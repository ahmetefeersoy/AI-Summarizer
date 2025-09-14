import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

from models import Note, NoteStatus
from ai_model import ai_model
from dotenv import load_dotenv

load_dotenv()

class BackgroundJobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.running = False
        
    async def start(self):
        self.running = True
        asyncio.create_task(self._process_jobs())
        
    async def stop(self):
        self.running = False
        
    async def add_job(self, job_id: str, job_type: str, data: Dict[str, Any]):
        self.jobs[job_id] = {
            "id": job_id,
            "type": job_type,
            "data": data,
            "status": "queued",
            "created_at": datetime.utcnow(),
            "attempts": 0,
            "max_attempts": 3
        }
        
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        return self.jobs.get(job_id, {"status": "not_found"})
        
    async def _process_jobs(self):
        
        while self.running:
            try:
                queued_jobs = [job for job in self.jobs.values() if job["status"] == "queued"]
                
                for job in queued_jobs:
                    if job["attempts"] >= job["max_attempts"]:
                        job["status"] = "failed"
                        await self._update_note_status(job["data"]["note_id"], "FAILED")
                        continue
                        
                    job["status"] = "processing"
                    job["attempts"] += 1
                    
                    try:
                        if job["type"] == "summarize_note":
                            await self._process_summarize_job(job)
                    except Exception as e:
                        logging.error(f"Error processing job {job['id']}: {str(e)}")
                        if job["attempts"] >= job["max_attempts"]:
                            job["status"] = "failed"
                            await self._update_note_status(job["data"]["note_id"], "FAILED")
                        else:
                            job["status"] = "queued"  
                            
                await asyncio.sleep(5)  
                
            except Exception as e:
                logging.error(f"Error in job processor: {str(e)}")
                await asyncio.sleep(5)
        
    async def _process_summarize_job(self, job: Dict[str, Any]):
        note_id = job["data"]["note_id"]
        raw_text = job["data"]["raw_text"]
        
        await self._update_note_status(note_id, "PROCESSING")
        
        summary = await ai_model.summarize_text(raw_text)
        
        note = await Note.get(id=note_id)
        note.summary = summary
        note.status = NoteStatus.DONE 
        await note.save()
        
        job["status"] = "completed"
        
    async def _update_note_status(self, note_id: int, status: str):
        note = await Note.get(id=note_id)
        note.status = NoteStatus(status)  
        await note.save()

job_manager = BackgroundJobManager()
