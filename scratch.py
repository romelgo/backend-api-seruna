import asyncio
import os
import sys

from routers.identification import _bg_send_whatsapp_fastapi

async def main():
    with open("test.jpg", "wb") as f:
        f.write(b"fake image data")
    
    # Needs a real student code that exists. Let's use the DB.
    # First get a student id.
    from supabase import create_client
    supa_url = os.getenv("SUPABASE_URL")
    supa_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    client = create_client(supa_url, supa_key)
    res = client.table("students").select("student_code, first_name").limit(1).execute()
    if not res.data:
        print("No students found")
        sys.exit(1)
        
    student = res.data[0]
    
    print(f"Testing with {student['student_code']}, {student['first_name']}")
    with open("test.jpg", "rb") as f:
        data = f.read()

    await _bg_send_whatsapp_fastapi(student['student_code'], student['first_name'], data)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())
