# Deploying to Railway

The app is deployed as **one Railway service**. The Docker image builds the React
frontend and FastAPI serves it, so the website and the API share one URL.
You don't need any CORS settings.

Files used for the deploy: `Dockerfile`, `.dockerignore`, `railway.json`, `requirements.txt`.

## 1. Push the code to a PRIVATE GitHub repo

The two model files (`models/car_parts.pt`, `models/car_damage.pt`) are committed
because Railway needs them to build the image. Keep the repository **private**.

1. Go to https://github.com/new, give it a name, choose **Private**, and click **Create repository**.
   Do not add a README or .gitignore.
2. In PowerShell, in the project folder:

```powershell
git commit -m "Prepare Railway deployment"
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## 2. Create the Railway project

1. Go to https://railway.com and sign in with GitHub.
2. Pick the **Hobby** plan ($5/month). Model inference needs about 2 GB of RAM, and
   the free trial's memory limit is too small, so video processing would crash.
3. Click **New Project → Deploy from GitHub repo**. Allow Railway to access the repo, then select it.
4. Railway reads `railway.json` and builds from the `Dockerfile`. The first build takes about 5–10 minutes.
   Watch it under **Deployments → View logs**.

## 3. Get a public URL

1. Open the service, then **Settings → Networking → Generate Domain**.
2. If Railway asks for a port, accept the one it suggests (usually 8080). The app listens on
   the `PORT` that Railway provides.
3. Open `https://<your-domain>/health`. You should see
   `"car_parts_checkpoint_found": true, "damage_checkpoint_found": true`.
4. Open `https://<your-domain>/` and upload a video.

## 4. Variables (optional)

You don't need to set any variables. The defaults in `app/config.py` match the local `.env`.
To change a setting, add it under the service's **Variables** tab. Railway redeploys
automatically. Useful ones:

| Variable | Default | Why change it |
|---|---|---|
| `SAMPLE_FPS` | `2.0` | Lower it (e.g. `1.0`) to process videos faster on CPU |
| `MAX_UPLOAD_MB` | `500` | Limit upload size |
| `DAMAGE_CONF`, `PARTS_CONF`, ... | see `.env` | Tune detection |

## 5. Updating the app later

```powershell
git add -A
git commit -m "Describe your change"
git push
```

Railway redeploys automatically on every push to `main`.

## Things to know

- **CPU only.** Railway has no GPU, so processing is slower than on a GPU machine. Expect a
  1-minute video to take several minutes.
- **One video at a time.** Other uploads wait in the queue.
- **Job history is in memory.** A redeploy or restart clears the job list, so a user who refreshes
  after a restart sees "Job not found". Uploaded videos and results are stored on the
  container disk, which is also cleared on redeploy.
- **Optional volume.** To keep result files across redeploys, right-click the service,
  choose **Attach Volume**, and set the mount path to `/app/data`. Uploads are never deleted
  automatically, so watch the disk usage.
- **Logs.** Errors appear under **Deployments → View logs**.
