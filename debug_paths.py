from config import ENTITIES_ANNO_DIR
from pathlib import Path

# Check a few XML filenames
xmls = list(ENTITIES_ANNO_DIR.glob("*.xml"))[:5]
print("Sample XMLs:", [x.name for x in xmls])
print("ANNO DIR exists:", ENTITIES_ANNO_DIR.exists())

# Simulate what dataset does for first HF row
from datasets import load_dataset
ds = load_dataset('nlphuji/flickr30k', split='test', revision='refs/convert/parquet')
train = [r for r in ds if r['split'] == 'train'][:5]
for r in train:
    flickr_id = r['filename'].replace('.jpg', '')
    xml_path = ENTITIES_ANNO_DIR / f"{flickr_id}.xml"
    print(f"filename={r['filename']} -> flickr_id={flickr_id} -> xml exists={xml_path.exists()}")
