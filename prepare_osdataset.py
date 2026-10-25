"""Convert the author's OSdataset ZIP into the paired C2RF directory layout."""
import argparse
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

from PIL import Image, ImageDraw


def prepare(archive, output_root, dataset_name, resolution, split, limit):
    groups = defaultdict(dict)
    pattern = re.compile(r'(opt|sar)(\d+)\.(png|jpg|jpeg|bmp|tif|tiff)', re.I)
    with ZipFile(archive) as source:
        for info in source.infolist():
            parts = PurePosixPath(info.filename).parts
            if len(parts) < 3 or parts[-3:-1] != (str(resolution), split):
                continue
            match = pattern.fullmatch(parts[-1])
            if match:
                modality, pair_id = match.group(1).lower(), int(match.group(2))
                if modality in groups[pair_id]:
                    raise ValueError('Duplicate %s image for pair %d' % (modality, pair_id))
                groups[pair_id][modality] = info
        incomplete = [key for key, pair in groups.items() if set(pair) != {'opt', 'sar'}]
        if incomplete:
            raise ValueError('Unpaired optical/SAR IDs: %s' % incomplete[:20])
        ids = sorted(groups)
        if not ids:
            raise ValueError('No opt<ID>/sar<ID> pairs found for %s/%s' % (resolution, split))
        selected = ids[:limit] if limit else ids
        root = Path(output_root) / split / dataset_name
        if (root / 'manifest.json').exists():
            raise FileExistsError('Prepared dataset already exists: %s' % root)
        images = []
        for pair_id in selected:
            record = {'id': pair_id, 'images': {}}
            for modality, folder in [('opt', 'vi'), ('sar', 'ir')]:
                info = groups[pair_id][modality]
                target = root / folder / ('%d.png' % pair_id)
                if target.exists():
                    raise FileExistsError('Refusing to overwrite %s' % target)
                with Image.open(BytesIO(source.read(info))) as img:
                    img.load()
                    if img.size != (resolution, resolution):
                        raise ValueError('Unexpected image size for %s: %s' % (info.filename, img.size))
                    record['images'][modality] = {
                        'source': info.filename, 'mode': img.mode,
                        'size': list(img.size), 'output': str(target),
                    }
                    target.parent.mkdir(parents=True, exist_ok=True)
                    img.save(target, format='PNG')
            images.append(record)
        manifest = {
            'source': 'https://github.com/xym2009/OSdataset',
            'archive': str(archive), 'resolution': resolution, 'split': split,
            'available_pairs': len(ids), 'prepared_pairs': len(selected),
            'images': images,
        }
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        preview_ids = selected[:4]
        tile = min(resolution, 256)
        preview = Image.new('RGB', (2 * tile, len(preview_ids) * (tile + 24)), 'white')
        draw = ImageDraw.Draw(preview)
        for row, pair_id in enumerate(preview_ids):
            for column, (folder, label) in enumerate([('vi', 'Optical'), ('ir', 'SAR')]):
                left, top = column * tile, row * (tile + 24)
                draw.text((left + 4, top + 4), '%s / %d' % (label, pair_id), fill='black')
                with Image.open(root / folder / ('%d.png' % pair_id)) as img:
                    preview.paste(img.convert('RGB').resize((tile, tile)), (left, top + 24))
        preview.save(root / 'preview.png')
    print('Available pairs: %d; prepared pairs: %d' % (len(ids), len(selected)))
    print('C2RF dataset: %s' % root)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('downloads/OSdataset.zip'))
    parser.add_argument('--output_root', type=Path, default=Path('dataset'))
    parser.add_argument('--dataset_name', default='OSDataset')
    parser.add_argument('--resolution', type=int, choices=[256, 512], default=256)
    parser.add_argument('--split', choices=['train', 'test', 'val'], default='test')
    parser.add_argument('--limit', type=int, default=0, help='0 prepares the whole selected split')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be non-negative')
    prepare(args.archive, args.output_root, args.dataset_name, args.resolution, args.split, args.limit)
