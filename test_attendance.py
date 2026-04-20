import asyncio
import httpx

async def test():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:3000/api/attendance",
            json={
                "student_code": "EST-2024001",
                "name": "Test Student",
                "confidence": 0.99
            },
            headers={"x-ai-server-secret": "face-control-secret-2024"}
        )
        print("Status", resp.status_code)
        print("Body", resp.text)

asyncio.run(test())
