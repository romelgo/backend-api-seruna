import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supa_url = os.getenv("SUPABASE_URL")
supa_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
client = create_client(supa_url, supa_key)

# We want to check if there are files ending in _py.jpg
# However, storage doesn't easy have a list-all-recursive.
# We can just get one student.
res = client.table("students").select("student_code, id").limit(5).execute()
for r in res.data:
    student_id = r["id"]
    files = client.storage.from_("attendance-photos").list(f"attendance/{student_id}")
    for f in files:
        if f["name"].endswith("_py.jpg"):
            print(f"Found _py.jpg uploaded for {student_id}: {f['name']}")

print("Done listing.")
