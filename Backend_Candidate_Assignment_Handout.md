# Backend Candidate Assignment – Handout

## Overview
We build AI-powered mobile apps and custom CRMs.  
Your task is to implement a small REST API with authentication, a background “AI summarize” job, SQL storage, and Docker deployment. Then record a short Loom video to explain your work.

---

## Tech Options
- Python (FastAPI) or Java (Spring Boot)  
- SQL database (Postgres/MySQL/SQLite)  
- Docker for containerization  

---

## Core Requirements
1. **Auth & Tenancy**  
   - Email/password signup/login.  
   - JWT (or similar).  
   - Roles: `ADMIN` and `AGENT`.  
   - Agents see only their own data; Admins see all.  

2. **Models**  
   - `users`  
   - `notes` (raw_text, summary, status, timestamps).  
   - Include DB migrations.  

3. **Async “AI summarize”**  
   - `POST /notes`: queues a job (stub/rule-based is fine).  
   - `GET /notes/{id}`: shows status (`queued|processing|done|failed`) and summary when ready.  
   - Include retries/idempotency if possible.  

4. **Optional Features (not required)**  
   - API polish: pagination, filtering, error handling, Swagger docs.  
   - CI/CD pipelines (GitHub Actions, etc.).  
   - Automated tests.  

---

## Dockerization & Deployment
- Provide a **Dockerfile**.  
- Deploy the container to a free cloud service (e.g., **Koyeb Hobby**).  
- Use a free Postgres option (Koyeb DB or Neon).  
- Keep secrets in environment variables.  

---

## Loom Video (max 10 minutes)
Please cover:
1. Live demo: signup/login, create note (queues summarize), check status.  
2. Code walkthrough: structure, auth, background job.  
3. Deployment: how and where you deployed it.  
4. Design choices: why you chose your approach.  

---

## Submission Checklist
- GitHub repo link (public).  
- Live API URL.  
- Loom video link.  

---

## Evaluation
- Auth & tenancy – 20  
- SQL modeling & migrations – 15  
- Async job & background processing – 20  
- Dockerization & deployment – 20  
- Loom clarity – 25  
- (Optional features may add bonus points)  

---

## Notes
- No UI is required.  
- Optional features are welcome but not mandatory.  
- Focus on clarity and working endpoints.  
