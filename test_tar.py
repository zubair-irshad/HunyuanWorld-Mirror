import tarfile

tar_path = "/home/mirshad7/Downloads/data/Omni6DPose/SOPE_webdataset/train/shard-000000.tar"

with tarfile.open(tar_path, "r") as tar:
    members = tar.getnames()

print("Total files in shard:", len(members))
for name in members[:40]:  # print first 40 entries for preview
    print(name)