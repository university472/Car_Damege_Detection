# Car Scope — React frontend

## Requirements

- Node.js 20+ (Node 22 recommended)
- The FastAPI backend running on `127.0.0.1:8000`

## Start

From this folder:

```powershell
npm install
npm run dev
```

Open <http://127.0.0.1:5173>.

The Vite development server proxies `/health`, `/api`, and `/artifacts` to the local FastAPI server at `http://127.0.0.1:8000`.

For an API hosted at a different origin, create a `.env` file in this folder containing:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Then configure the backend's `CORS_ORIGINS` to include the frontend origin. For example, the local development origins are `http://127.0.0.1:5173` and `http://localhost:5173`.

## Result table

The result table contains only two columns:

- **Damaged part name** — the predicted car-part name, displayed with spaces.
- **Number of damaged parts** — the number of reported part instances for that name.

The count is based on unique part-instance records in `report.damaged_parts`, grouped by `part_name`. It is not the count of damage regions. Unassigned damage regions remain review items in the API's full JSON report and are not guessed into a car-part category.

## Evidence image gallery

After a video analysis completes, the page loads every supported image saved under that job's `evidence_frames/` folder. Click a thumbnail to open a larger preview. The preview also includes a link to download an individual image.

Use **Download all images** to download a ZIP archive containing all evidence images in a folder named `evidence_frames/`.

The frontend uses these backend endpoints:

- `GET /api/jobs/{job_id}/evidence-frames` — returns image names, artifact URLs, frame numbers, timestamps and file sizes.
- `GET /api/jobs/{job_id}/evidence-frames/download` — creates and downloads a ZIP archive of all image files in that job's evidence folder.
- `GET /artifacts/{job_id}/evidence_frames/{filename}` — serves an evidence image for the gallery.

The pipeline currently creates annotated evidence frames when a sampled frame has at least one damage prediction. If the folder is empty, the gallery explains that no evidence images were generated. The original upload and two-column damaged-parts summary table are unchanged.
