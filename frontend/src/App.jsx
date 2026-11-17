// import { useEffect, useMemo, useRef, useState } from 'react'

// const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
// const MAX_UPLOAD_BYTES = 500 * 1024 * 1024
// const ALLOWED_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v']

// function formatBytes(bytes) {
//   if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'
//   const units = ['B', 'KB', 'MB', 'GB']
//   const index = Math.min(
//     Math.floor(Math.log(bytes) / Math.log(1024)),
//     units.length - 1
//   )
//   return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`
// }

// function getPartRows(report) {
//   const summary = report?.summary || {}
//   const hasV21PartSummary = Array.isArray(summary.confirmed_parts_by_name)

//   // Section 2 shows ONLY confirmed damaged parts. Review candidates are
//   // displayed in their own dedicated review panel further down the page.
//   if (hasV21PartSummary) {
//     const rows = []
//     for (const item of summary.confirmed_parts_by_name || []) {
//       const name = String(item?.part_name || '').trim()
//       const count = Math.max(0, Number(item?.count) || 0)
//       if (!name || count === 0) continue
//       rows.push({
//         partName: name,
//         count,
//         confirmedCount: count,
//         reviewCount: 0
//       })
//     }
//     return rows.sort((a, b) => a.partName.localeCompare(b.partName))
//   }

//   // Backward compatibility for reports generated before V2.1.
//   const instances = new Map()
//   let fallbackIndex = 0

//   for (const part of report?.damaged_parts || []) {
//     const name = String(part?.part_name || '').trim()
//     const regions = Array.isArray(part?.damage_regions)
//       ? part.damage_regions
//       : []
//     if (!name || regions.length === 0) continue

//     const suppliedId = String(part?.part_instance_id || '').trim()
//     const instanceId = suppliedId || `fallback:${name}:${fallbackIndex++}`
//     const confirmed =
//       part?.damage_status === 'detected' ||
//       regions.some((region) => region?.damage_status === 'detected')

//     if (confirmed) instances.set(instanceId, { partName: name })
//   }

//   const counts = new Map()
//   for (const instance of instances.values()) {
//     const row = counts.get(instance.partName) || {
//       partName: instance.partName,
//       count: 0,
//       confirmedCount: 0,
//       reviewCount: 0
//     }
//     row.count += 1
//     row.confirmedCount += 1
//     counts.set(instance.partName, row)
//   }

//   return [...counts.values()].sort((a, b) =>
//     a.partName.localeCompare(b.partName)
//   )
// }

// function humanizeReason(reason) {
//   return String(reason || '')
//     .replaceAll('_', ' ')
//     .replace(/\s+/g, ' ')
//     .trim()
// }

// function getArtifactUrl(jobId, relativePath) {
//   if (!jobId || !relativePath) return null
//   const safePath = String(relativePath)
//     .replaceAll('\\', '/')
//     .split('/')
//     .filter(Boolean)
//     .map((part) => encodeURIComponent(part))
//     .join('/')
//   return `${API_BASE}/artifacts/${encodeURIComponent(jobId)}/${safePath}`
// }

// function formatTimestamp(seconds) {
//   if (!Number.isFinite(seconds) || seconds < 0) return 'Timestamp unavailable'
//   const minutes = Math.floor(seconds / 60)
//   const remainingSeconds = seconds - minutes * 60
//   return `${minutes}:${remainingSeconds.toFixed(2).padStart(5, '0')}`
// }

// function friendlyError(message) {
//   if (!message) return 'Something went wrong while processing the video.'
//   try {
//     const parsed = JSON.parse(message)
//     return parsed.detail || message
//   } catch {
//     return message
//   }
// }

// function App() {
//   const inputRef = useRef(null)
//   const [selectedFile, setSelectedFile] = useState(null)
//   const [dragging, setDragging] = useState(false)
//   const [uploading, setUploading] = useState(false)
//   const [jobId, setJobId] = useState('')
//   const [jobStatus, setJobStatus] = useState('idle')
//   const [report, setReport] = useState(null)
//   const [evidenceFrames, setEvidenceFrames] = useState([])
//   const [evidenceLoading, setEvidenceLoading] = useState(false)
//   const [evidenceError, setEvidenceError] = useState('')
//   const [selectedEvidence, setSelectedEvidence] = useState(null)
//   const [error, setError] = useState('')
//   const [apiHealth, setApiHealth] = useState({
//     state: 'checking',
//     message: 'Connecting to analysis service…'
//   })

//   useEffect(() => {
//     let cancelled = false
//     fetch(`${API_BASE}/health`)
//       .then(async (response) => {
//         const data = await response.json().catch(() => ({}))
//         if (!response.ok)
//           throw new Error(
//             data.detail || 'The analysis service is not responding.'
//           )
//         if (cancelled) return
//         const checkpointsReady =
//           data.car_parts_checkpoint_found && data.damage_checkpoint_found
//         setApiHealth({
//           state: checkpointsReady ? 'ready' : 'warning',
//           message: checkpointsReady
//             ? 'Analysis service connected · both model checkpoints found'
//             : 'API connected, but one or both model checkpoints are missing'
//         })
//       })
//       .catch((err) => {
//         if (!cancelled)
//           setApiHealth({
//             state: 'offline',
//             message: err.message || 'Cannot connect to the analysis API'
//           })
//       })
//     return () => {
//       cancelled = true
//     }
//   }, [])

//   useEffect(() => {
//     if (!jobId) return undefined
//     let cancelled = false
//     let timer

//     const poll = async () => {
//       try {
//         const response = await fetch(`${API_BASE}/api/jobs/${jobId}`)
//         const data = await response.json().catch(() => ({}))
//         if (!response.ok)
//           throw new Error(data.detail || 'Could not check analysis status.')
//         if (cancelled) return

//         if (data.status === 'failed') {
//           setJobStatus('failed')
//           setError(
//             data.error ||
//               'Video analysis failed. Check the backend terminal for details.'
//           )
//           return
//         }
//         if (data.status === 'completed') {
//           setJobStatus('completed')
//           const reportResponse = await fetch(
//             `${API_BASE}/api/jobs/${jobId}/report`
//           )
//           const reportData = await reportResponse.json().catch(() => ({}))
//           if (!reportResponse.ok)
//             throw new Error(
//               reportData.detail ||
//                 'Analysis finished, but the report could not be loaded.'
//             )
//           if (!cancelled) setReport(reportData)
//           return
//         }
//         setJobStatus(data.status)
//         timer = window.setTimeout(poll, 1800)
//       } catch (err) {
//         if (cancelled) return
//         setError(err.message || 'Could not communicate with the analysis API.')
//         setJobStatus('failed')
//       }
//     }

//     poll()
//     return () => {
//       cancelled = true
//       window.clearTimeout(timer)
//     }
//   }, [jobId])

//   useEffect(() => {
//     if (!jobId || jobStatus !== 'completed' || !report) return undefined
//     let cancelled = false
//     setEvidenceLoading(true)
//     setEvidenceError('')

//     fetch(`${API_BASE}/api/jobs/${jobId}/evidence-frames`)
//       .then(async (response) => {
//         const data = await response.json().catch(() => ({}))
//         if (!response.ok)
//           throw new Error(
//             data.detail || 'Could not load evidence frame images.'
//           )
//         return data
//       })
//       .then((data) => {
//         if (!cancelled)
//           setEvidenceFrames(
//             Array.isArray(data.evidence_frames) ? data.evidence_frames : []
//           )
//       })
//       .catch((err) => {
//         if (cancelled) return
//         setEvidenceFrames([])
//         setEvidenceError(err.message || 'Could not load evidence frame images.')
//       })
//       .finally(() => {
//         if (!cancelled) setEvidenceLoading(false)
//       })

//     return () => {
//       cancelled = true
//     }
//   }, [jobId, jobStatus, report])

//   useEffect(() => {
//     if (!selectedEvidence) return undefined
//     const onKeyDown = (event) => {
//       if (event.key === 'Escape') setSelectedEvidence(null)
//     }
//     window.addEventListener('keydown', onKeyDown)
//     return () => window.removeEventListener('keydown', onKeyDown)
//   }, [selectedEvidence])

//   const partRows = useMemo(() => getPartRows(report), [report])
//   const confirmedPartInstances = Number(
//     report?.summary?.confirmed_part_instances ??
//       partRows.reduce((sum, row) => sum + row.confirmedCount, 0)
//   )
//   // Review candidates are informational only and never contribute to the
//   // damaged-part count shown in Section 2.
//   const reviewPartInstances = Number(
//     report?.summary?.review_part_instances ?? 0
//   )
//   const totalDamagedParts = confirmedPartInstances
//   const reviewCandidateTracks = Array.isArray(report?.review_candidates)
//     ? report.review_candidates.length
//     : 0
//   const unassignedDamageTracks = Number(
//     report?.summary?.unassigned_damage_regions ??
//       report?.unassigned_damage_regions?.length ??
//       0
//   )
//   const confirmedDamageTracks = Number(
//     report?.summary?.confirmed_damage_regions ?? 0
//   )

//   const setFile = (file) => {
//     setError('')
//     setReport(null)
//     setEvidenceFrames([])
//     setEvidenceLoading(false)
//     setEvidenceError('')
//     setSelectedEvidence(null)
//     setJobId('')
//     setJobStatus('idle')
//     if (!file) {
//       setSelectedFile(null)
//       return
//     }
//     const extension = `.${(file.name.split('.').pop() || '').toLowerCase()}`
//     if (!ALLOWED_EXTENSIONS.includes(extension)) {
//       setSelectedFile(null)
//       setError(
//         'Choose a supported video file: MP4, MOV, AVI, MKV, WEBM or M4V.'
//       )
//       return
//     }
//     if (file.size > MAX_UPLOAD_BYTES) {
//       setSelectedFile(null)
//       setError('This video is larger than the 500 MB upload limit.')
//       return
//     }
//     if (file.size === 0) {
//       setSelectedFile(null)
//       setError('The selected file is empty. Please choose another video.')
//       return
//     }
//     setSelectedFile(file)
//   }

//   const handleUpload = async (event) => {
//     event.preventDefault()
//     if (
//       !selectedFile ||
//       uploading ||
//       ['queued', 'processing'].includes(jobStatus)
//     )
//       return

//     setUploading(true)
//     setError('')
//     setReport(null)
//     setEvidenceFrames([])
//     setEvidenceLoading(false)
//     setEvidenceError('')
//     setSelectedEvidence(null)
//     setJobId('')
//     setJobStatus('idle')
//     try {
//       const formData = new FormData()
//       formData.append('video', selectedFile)
//       const response = await fetch(`${API_BASE}/api/jobs`, {
//         method: 'POST',
//         body: formData
//       })
//       const data = await response.json().catch(() => ({}))
//       if (!response.ok) throw new Error(data.detail || 'Video upload failed.')
//       setJobId(data.job_id)
//       setJobStatus(data.status || 'queued')
//     } catch (err) {
//       setError(
//         err.message ||
//           'Could not upload the video. Make sure the backend is running.'
//       )
//     } finally {
//       setUploading(false)
//     }
//   }

//   const resetAnalysis = () => {
//     setSelectedFile(null)
//     setJobId('')
//     setJobStatus('idle')
//     setReport(null)
//     setEvidenceFrames([])
//     setEvidenceLoading(false)
//     setEvidenceError('')
//     setSelectedEvidence(null)
//     setError('')
//     if (inputRef.current) inputRef.current.value = ''
//   }

//   const busy = uploading || ['queued', 'processing'].includes(jobStatus)

//   return (
//     <div className="app-shell">
//       <header className="topbar">
//         <a className="brand" href="#top" aria-label="Car Damage Analyzer home">
//           <span className="brand-mark" aria-hidden="true">
//             <svg viewBox="0 0 32 32" fill="none">
//               <path
//                 d="M5 19.5 7.7 12a3 3 0 0 1 2.8-2h11a3 3 0 0 1 2.8 2l2.7 7.5v5H5v-5Z"
//                 stroke="currentColor"
//                 strokeWidth="1.8"
//                 strokeLinejoin="round"
//               />
//               <path
//                 d="m9 12.2 1.4 5h11.2l1.4-5M5.5 20h21M9 24.5v2m14-2v2"
//                 stroke="currentColor"
//                 strokeWidth="1.8"
//                 strokeLinecap="round"
//               />
//               <circle cx="10" cy="20.5" r="1" fill="currentColor" />
//               <circle cx="22" cy="20.5" r="1" fill="currentColor" />
//             </svg>
//           </span>
//           <span className="brand-copy">
//             <strong>CAR SCOPE</strong>
//             <small>VIDEO DAMAGE ANALYSIS</small>
//           </span>
//         </a>
//         <div
//           className={`service-status ${apiHealth.state}`}
//           title={apiHealth.message}
//         >
//           <span className="status-dot" />
//           <span>
//             {apiHealth.state === 'ready'
//               ? 'System ready'
//               : apiHealth.state === 'checking'
//                 ? 'Connecting'
//                 : apiHealth.state === 'warning'
//                   ? 'Model check needed'
//                   : 'API unavailable'}
//           </span>
//         </div>
//       </header>

//       <main id="top" className="main-content">
//         <section className="intro-section">
//           <div className="eyebrow">
//             <span className="eyebrow-line" /> AI-POWERED VEHICLE INSPECTION
//           </div>
//           <h1>
//             See the damage.
//             <br />
//             <span>Know the parts.</span>
//           </h1>
//           <p className="intro-copy">
//             Upload a walk-around video of a vehicle. The analysis pipeline scans
//             frames, identifies affected car parts, and summarizes the results
//             for you.
//           </p>
//         </section>

//         <section className="workspace-grid">
//           <form className="upload-card panel" onSubmit={handleUpload}>
//             <div className="section-heading">
//               <div>
//                 <span className="step-label">STEP 01</span>
//                 <h2>Upload video</h2>
//               </div>
//               <span className="file-type-tag">MP4 · MOV · AVI</span>
//             </div>

//             <div
//               className={`dropzone ${dragging ? 'dragging' : ''} ${selectedFile ? 'has-file' : ''}`}
//               onDragOver={(event) => {
//                 event.preventDefault()
//                 setDragging(true)
//               }}
//               onDragLeave={() => setDragging(false)}
//               onDrop={(event) => {
//                 event.preventDefault()
//                 setDragging(false)
//                 setFile(event.dataTransfer.files?.[0])
//               }}
//               role="button"
//               tabIndex={0}
//               onKeyDown={(event) => {
//                 if (event.key === 'Enter' || event.key === ' ') {
//                   event.preventDefault()
//                   inputRef.current?.click()
//                 }
//               }}
//               onClick={(event) => {
//                 if (event.target !== inputRef.current) inputRef.current?.click()
//               }}
//               aria-label="Choose a video file or drop it here"
//             >
//               <input
//                 ref={inputRef}
//                 className="hidden-input"
//                 type="file"
//                 accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,video/x-m4v,.mp4,.mov,.avi,.mkv,.webm,.m4v"
//                 onChange={(event) => setFile(event.target.files?.[0])}
//               />
//               <div className="upload-icon-wrap">
//                 {selectedFile ? (
//                   <svg viewBox="0 0 24 24" fill="none">
//                     <path
//                       d="M5 12.5 9.2 17 19 7"
//                       stroke="currentColor"
//                       strokeWidth="1.8"
//                       strokeLinecap="round"
//                       strokeLinejoin="round"
//                     />
//                   </svg>
//                 ) : (
//                   <svg viewBox="0 0 24 24" fill="none">
//                     <path
//                       d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4 16.5v2A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5v-2"
//                       stroke="currentColor"
//                       strokeWidth="1.7"
//                       strokeLinecap="round"
//                       strokeLinejoin="round"
//                     />
//                   </svg>
//                 )}
//               </div>
//               {selectedFile ? (
//                 <>
//                   <strong className="file-name">{selectedFile.name}</strong>
//                   <span className="drop-hint">
//                     {formatBytes(selectedFile.size)} · Ready to analyze
//                   </span>
//                 </>
//               ) : (
//                 <>
//                   <strong>Drop your video here</strong>
//                   <span className="drop-hint">
//                     or <span className="browse-link">browse files</span>
//                   </span>
//                   <span className="drop-footnote">
//                     Maximum file size: 500 MB
//                   </span>
//                 </>
//               )}
//             </div>

//             {error && (
//               <div className="error-message" role="alert">
//                 <span className="error-symbol">!</span>
//                 <span>{friendlyError(error)}</span>
//               </div>
//             )}

//             <button
//               className="primary-button"
//               type="submit"
//               disabled={!selectedFile || busy || apiHealth.state === 'offline'}
//             >
//               {uploading ? (
//                 <>
//                   <span className="button-spinner" /> Uploading video…
//                 </>
//               ) : busy ? (
//                 <>
//                   <span className="button-spinner" /> Analyzing video…
//                 </>
//               ) : (
//                 <>
//                   Analyze video <span aria-hidden="true">↗</span>
//                 </>
//               )}
//             </button>
//             <p className="privacy-note">
//               <span className="lock-icon" aria-hidden="true">
//                 ⌑
//               </span>{' '}
//               Your video is processed by the connected analysis server.
//             </p>
//           </form>

//           <section
//             className="results-card panel"
//             aria-labelledby="results-title"
//           >
//             <div className="section-heading results-heading">
//               <div>
//                 <span className="step-label">STEP 02</span>
//                 <h2 id="results-title">Detection summary</h2>
//               </div>
//               {report && (
//                 <span className="complete-tag">
//                   <span /> Complete
//                 </span>
//               )}
//             </div>

//             {!report && !busy && jobStatus !== 'failed' && (
//               <div className="empty-results">
//                 <div className="empty-illustration" aria-hidden="true">
//                   <svg viewBox="0 0 150 112" fill="none">
//                     <rect
//                       x="20"
//                       y="15"
//                       width="110"
//                       height="78"
//                       rx="8"
//                       stroke="currentColor"
//                       strokeWidth="1.5"
//                       strokeDasharray="4 5"
//                     />
//                     <path
//                       d="m35 72 20-21 15 14 13-16 31 30"
//                       stroke="currentColor"
//                       strokeWidth="1.5"
//                       strokeLinecap="round"
//                       strokeLinejoin="round"
//                     />
//                     <circle
//                       cx="100"
//                       cy="34"
//                       r="7"
//                       stroke="currentColor"
//                       strokeWidth="1.5"
//                     />
//                     <path
//                       d="M54 100h42"
//                       stroke="currentColor"
//                       strokeWidth="1.5"
//                       strokeLinecap="round"
//                     />
//                   </svg>
//                 </div>
//                 <h3>Your results will appear here</h3>
//                 <p>
//                   Upload a vehicle video to see the affected part names and
//                   counts.
//                 </p>
//                 <div className="table-preview" aria-hidden="true">
//                   <div>
//                     <span>Damaged part name</span>
//                     <span>Count</span>
//                   </div>
//                   <div>
//                     <i />
//                     <i />
//                   </div>
//                   <div>
//                     <i />
//                     <i />
//                   </div>
//                 </div>
//               </div>
//             )}

//             {busy && (
//               <div className="processing-state" aria-live="polite">
//                 <div className="processing-orbit">
//                   <span />
//                   <span />
//                   <span />
//                 </div>
//                 <h3>
//                   {jobStatus === 'queued'
//                     ? 'Your video is in the queue'
//                     : 'Analyzing your video'}
//                 </h3>
//                 <p>
//                   {jobStatus === 'queued'
//                     ? 'The analysis service will start processing shortly.'
//                     : 'Both vision models are reviewing the sampled frames. Longer videos may take more time.'}
//                 </p>
//                 <div className="progress-track">
//                   <span className="progress-indeterminate" />
//                 </div>
//                 <span className="processing-caption">
//                   PLEASE KEEP THIS PAGE OPEN
//                 </span>
//               </div>
//             )}

//             {report && (
//               <div className="results-content">
//                 <div className="metric-row">
//                   <div className="metric-icon" aria-hidden="true">
//                     <svg viewBox="0 0 24 24" fill="none">
//                       <path
//                         d="M4 7.5 12 3l8 4.5v9L12 21l-8-4.5v-9Z"
//                         stroke="currentColor"
//                         strokeWidth="1.6"
//                         strokeLinejoin="round"
//                       />
//                       <path
//                         d="m4.5 7.8 7.5 4.4 7.5-4.4M12 12.5V21"
//                         stroke="currentColor"
//                         strokeWidth="1.6"
//                         strokeLinejoin="round"
//                       />
//                     </svg>
//                   </div>
//                   <div className="metric-copy">
//                     <span>Confirmed damaged parts</span>
//                     <strong>{totalDamagedParts}</strong>
//                   </div>
//                   <span className="metric-subtext">
//                     {reviewPartInstances > 0
//                       ? `${reviewPartInstances} additional part(s) pending review`
//                       : 'All detected damage is confirmed'}
//                   </span>
//                 </div>
//                 <div className="table-topline">
//                   <div>
//                     <span className="table-title">Confirmed damaged parts</span>
//                     <span className="table-subtitle">
//                       Grouped by part name · only parts with confirmed damage
//                       are shown
//                     </span>
//                   </div>
//                   <span className="row-count">
//                     {partRows.length} {partRows.length === 1 ? 'row' : 'rows'}
//                   </span>
//                 </div>
//                 <div className="table-wrap">
//                   <table>
//                     <thead>
//                       <tr>
//                         <th scope="col">Part name</th>
//                         <th scope="col" className="count-column">
//                           Part instances
//                         </th>
//                       </tr>
//                     </thead>
//                     <tbody>
//                       {partRows.length ? (
//                         partRows.map((row) => (
//                           <tr key={row.partName}>
//                             <td>
//                               <span className="part-cell">
//                                 <span className="part-dot" />
//                                 {row.partName.replaceAll('_', ' ')}
//                               </span>
//                               {row.reviewCount > 0 && (
//                                 <span
//                                   title={`${row.reviewCount} part instance(s) need review`}
//                                   style={{
//                                     display: 'inline-block',
//                                     margin: '5px 0 0 18px',
//                                     padding: '3px 7px',
//                                     borderRadius: '999px',
//                                     background: '#fff4df',
//                                     color: '#8a5a00',
//                                     fontSize: '11px',
//                                     lineHeight: 1.4,
//                                     fontWeight: 600
//                                   }}
//                                 >
//                                   {row.confirmedCount > 0
//                                     ? 'Some need review'
//                                     : 'Needs review'}
//                                 </span>
//                               )}
//                             </td>
//                             <td className="count-column">
//                               <span className="count-pill">{row.count}</span>
//                             </td>
//                           </tr>
//                         ))
//                       ) : (
//                         <tr>
//                           <td colSpan="2" className="no-damage-cell">
//                             <span className="no-damage-icon">✓</span>
//                             <strong>
//                               No confirmed damaged parts were detected
//                             </strong>
//                             <span>
//                               {reviewPartInstances > 0
//                                 ? `${reviewPartInstances} part instance(s) still require manual review — see the review panel below.`
//                                 : `${unassignedDamageTracks} unassigned damage track(s) remain. Review the evidence frames below.`}
//                             </span>
//                           </td>
//                         </tr>
//                       )}
//                     </tbody>
//                   </table>
//                 </div>
//                 <div className="result-footer">
//                   <span>
//                     <span className="footer-dot" /> Report generated
//                   </span>
//                   <span>
//                     {report.video?.processed_frames ?? '—'} frames analyzed
//                   </span>
//                 </div>
//                 <button
//                   className="secondary-button"
//                   type="button"
//                   onClick={resetAnalysis}
//                 >
//                   Analyze another video <span aria-hidden="true">↗</span>
//                 </button>
//                 <div
//                   className="v21-report-stats"
//                   aria-label="V2.1 damage association summary"
//                 >
//                   <div>
//                     <strong>{confirmedDamageTracks}</strong>
//                     <span>confirmed damage tracks</span>
//                   </div>
//                   <div>
//                     <strong>{reviewCandidateTracks}</strong>
//                     <span>tracks with part candidates</span>
//                   </div>
//                   <div>
//                     <strong>{unassignedDamageTracks}</strong>
//                     <span>unassigned tracks</span>
//                   </div>
//                 </div>
//                 {(reviewPartInstances > 0 || unassignedDamageTracks > 0) && (
//                   <p className="review-note">
//                     V2.1 found {reviewPartInstances} named part instance(s) that
//                     require manual review and {unassignedDamageTracks}
//                     unassigned damage track(s). The {reviewCandidateTracks}
//                     review entries are damage tracks, not unique part counts.
//                     Evidence images show model predictions; they do not by
//                     themselves prove physical damage or a confirmed part
//                     association.
//                   </p>
//                 )}
//                 {reviewCandidateTracks > 0 && (
//                   <details className="v21-review-details">
//                     <summary>
//                       Inspect {reviewCandidateTracks} damage tracks requiring
//                       association review
//                     </summary>
//                     <div className="v21-review-list">
//                       {report.review_candidates.map((candidate, index) => {
//                         const region =
//                           candidate?.damage_region || candidate || {}
//                         const candidateNames = Array.isArray(
//                           candidate?.candidate_part_names
//                         )
//                           ? candidate.candidate_part_names
//                           : Array.isArray(region?.candidate_part_names)
//                             ? region.candidate_part_names
//                             : []
//                         const primaryName =
//                           candidate?.candidate_part_name ||
//                           candidateNames[0] ||
//                           'No reliable part candidate'
//                         const reasons = Array.isArray(candidate?.review_reasons)
//                           ? candidate.review_reasons
//                           : Array.isArray(region?.review_reasons)
//                             ? region.review_reasons
//                             : []
//                         const representativePath =
//                           region?.representative_frame_path
//                         const evidenceUrl = getArtifactUrl(
//                           jobId,
//                           representativePath
//                         )
//                         const score = Number(region?.damage_confidence)
//                         const consistency = Number(
//                           candidate?.association_consistency ??
//                             region?.association_consistency
//                         )
//                         const frameNumber = region?.representative_frame
//                         const damageId =
//                           candidate?.damage_id ||
//                           region?.damage_id ||
//                           `review-${index + 1}`

//                         return (
//                           <article className="v21-review-item" key={damageId}>
//                             <div className="v21-review-item-heading">
//                               <div>
//                                 <strong>
//                                   {String(primaryName).replaceAll('_', ' ')}
//                                 </strong>
//                                 <span>{damageId}</span>
//                               </div>
//                               <span className="v21-review-badge">
//                                 Manual review
//                               </span>
//                             </div>
//                             <div className="v21-review-meta">
//                               <span>
//                                 Raw model score:{' '}
//                                 {Number.isFinite(score)
//                                   ? score.toFixed(3)
//                                   : '—'}
//                               </span>
//                               <span>
//                                 Association consistency:{' '}
//                                 {Number.isFinite(consistency)
//                                   ? `${(consistency * 100).toFixed(1)}%`
//                                   : '—'}
//                               </span>
//                               <span>
//                                 Representative frame:{' '}
//                                 {frameNumber != null ? frameNumber : '—'}
//                               </span>
//                               {evidenceUrl && (
//                                 <a
//                                   href={evidenceUrl}
//                                   target="_blank"
//                                   rel="noreferrer"
//                                 >
//                                   Open evidence frame ↗
//                                 </a>
//                               )}
//                             </div>
//                             {candidateNames.length > 1 && (
//                               <p className="v21-review-alternatives">
//                                 Other candidates:{' '}
//                                 {candidateNames
//                                   .filter((name) => name !== primaryName)
//                                   .map((name) =>
//                                     String(name).replaceAll('_', ' ')
//                                   )
//                                   .join(', ')}
//                               </p>
//                             )}
//                             {reasons.length > 0 && (
//                               <p className="v21-review-reasons">
//                                 {reasons
//                                   .slice(0, 4)
//                                   .map(humanizeReason)
//                                   .join(' · ')}
//                               </p>
//                             )}
//                           </article>
//                         )
//                       })}
//                     </div>
//                   </details>
//                 )}
//               </div>
//             )}

//             {jobStatus === 'failed' && !report && (
//               <div className="failed-state">
//                 <div className="failed-icon">!</div>
//                 <h3>Analysis could not be completed</h3>
//                 <p>{friendlyError(error)}</p>
//                 <button
//                   className="secondary-button"
//                   type="button"
//                   onClick={resetAnalysis}
//                 >
//                   Try another video
//                 </button>
//               </div>
//             )}
//           </section>
//         </section>

//         {report && (
//           <section
//             className="evidence-card panel"
//             aria-labelledby="evidence-title"
//           >
//             <div className="evidence-heading">
//               <div>
//                 <span className="step-label">STEP 03</span>
//                 <h2 id="evidence-title">Evidence frames</h2>
//                 <p className="evidence-description">
//                   Annotated frames containing model-predicted damage regions.
//                   Review these predictions before treating them as confirmed
//                   damage.
//                 </p>
//               </div>
//               <div className="evidence-heading-actions">
//                 <span className="row-count">
//                   {evidenceFrames.length}{' '}
//                   {evidenceFrames.length === 1 ? 'image' : 'images'}
//                 </span>
//                 {evidenceFrames.length > 0 && (
//                   <a
//                     className="download-evidence-button"
//                     href={`${API_BASE}/api/jobs/${jobId}/evidence-frames/download`}
//                     download
//                     title="Download all evidence images in one ZIP file"
//                   >
//                     <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
//                       <path
//                         d="M12 3v11m0 0 4-4m-4 4-4-4M4 17v3h16v-3"
//                         stroke="currentColor"
//                         strokeWidth="1.7"
//                         strokeLinecap="round"
//                         strokeLinejoin="round"
//                       />
//                     </svg>
//                     Download all images
//                   </a>
//                 )}
//               </div>
//             </div>

//             {evidenceLoading && (
//               <div className="evidence-state">
//                 <span className="button-spinner evidence-spinner" /> Loading
//                 evidence images…
//               </div>
//             )}
//             {!evidenceLoading && evidenceError && (
//               <div className="evidence-state evidence-error" role="alert">
//                 {evidenceError} Refresh the page or check that the backend is
//                 running.
//               </div>
//             )}
//             {!evidenceLoading &&
//               !evidenceError &&
//               evidenceFrames.length === 0 && (
//                 <div className="evidence-empty">
//                   <div className="evidence-empty-icon" aria-hidden="true">
//                     ▧
//                   </div>
//                   <strong>No evidence images were generated</strong>
//                   <span>
//                     The pipeline only saves evidence frames when it detects a
//                     damage region.
//                   </span>
//                 </div>
//               )}
//             {!evidenceLoading &&
//               !evidenceError &&
//               evidenceFrames.length > 0 && (
//                 <div className="evidence-grid">
//                   {evidenceFrames.map((frame) => {
//                     const imageUrl = `${API_BASE}${frame.url}`
//                     const timestamp = formatTimestamp(frame.timestamp_seconds)
//                     return (
//                       <button
//                         className="evidence-thumb"
//                         type="button"
//                         key={frame.filename}
//                         onClick={() => setSelectedEvidence(frame)}
//                         aria-label={`View evidence image for frame ${frame.frame_number ?? frame.filename}`}
//                       >
//                         <div className="evidence-image-wrap">
//                           <img
//                             src={imageUrl}
//                             alt={`Annotated car damage evidence at ${timestamp}`}
//                             loading="lazy"
//                           />
//                           <span className="evidence-view-chip">
//                             View image ↗
//                           </span>
//                         </div>
//                         <span className="evidence-thumb-meta">
//                           <strong>
//                             {frame.frame_number != null
//                               ? `Frame ${frame.frame_number}`
//                               : frame.filename}
//                           </strong>
//                           <span>{timestamp}</span>
//                         </span>
//                       </button>
//                     )
//                   })}
//                 </div>
//               )}
//           </section>
//         )}

//         {selectedEvidence && (
//           <div
//             className="evidence-modal-backdrop"
//             role="presentation"
//             onMouseDown={(event) => {
//               if (event.target === event.currentTarget)
//                 setSelectedEvidence(null)
//             }}
//           >
//             <section
//               className="evidence-modal"
//               role="dialog"
//               aria-modal="true"
//               aria-labelledby="evidence-modal-title"
//             >
//               <div className="evidence-modal-header">
//                 <div>
//                   <span className="step-label">ANNOTATED FRAME</span>
//                   <h2 id="evidence-modal-title">
//                     {selectedEvidence.frame_number != null
//                       ? `Frame ${selectedEvidence.frame_number}`
//                       : selectedEvidence.filename}
//                   </h2>
//                   <span className="evidence-modal-time">
//                     {formatTimestamp(selectedEvidence.timestamp_seconds)}
//                   </span>
//                 </div>
//                 <button
//                   className="modal-close-button"
//                   type="button"
//                   onClick={() => setSelectedEvidence(null)}
//                   aria-label="Close image preview"
//                 >
//                   ×
//                 </button>
//               </div>
//               <img
//                 className="evidence-modal-image"
//                 src={`${API_BASE}${selectedEvidence.url}`}
//                 alt={`Annotated evidence image ${selectedEvidence.filename}`}
//               />
//               <div className="evidence-modal-footer">
//                 <span>{selectedEvidence.filename}</span>
//                 <a
//                   href={`${API_BASE}${selectedEvidence.url}`}
//                   download={selectedEvidence.filename}
//                 >
//                   Download this image
//                 </a>
//               </div>
//             </section>
//           </div>
//         )}

//         <footer className="page-footer">
//           <span>
//             CAR SCOPE <span className="footer-divider">/</span> VIDEO INSPECTION
//           </span>
//           <span>AI-assisted results · Review important decisions manually</span>
//         </footer>
//       </main>
//     </div>
//   )
// }

// export default App

import { useEffect, useMemo, useRef, useState } from 'react'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
const MAX_UPLOAD_BYTES = 500 * 1024 * 1024
const ALLOWED_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v']

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const index = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1
  )
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`
}

function getPartRows(report) {
  const summary = report?.summary || {}
  const hasV21 =
    Array.isArray(summary.confirmed_parts_by_name) ||
    Array.isArray(summary.review_candidates_by_name)

  // Section 2 groups confirmed damaged parts AND review candidates by
  // part name. Rows with only review candidates are flagged with a
  // "Needs review" badge; rows with both are flagged "Some need review".
  if (hasV21) {
    const byName = new Map()
    const add = (name, count, kind) => {
      const trimmed = String(name || '').trim()
      const n = Math.max(0, Number(count) || 0)
      if (!trimmed || n === 0) return
      const row = byName.get(trimmed) || {
        partName: trimmed,
        confirmedCount: 0,
        reviewCount: 0,
        count: 0
      }
      row[kind] += n
      row.count += n
      byName.set(trimmed, row)
    }
    for (const item of summary.confirmed_parts_by_name || [])
      add(item?.part_name, item?.count, 'confirmedCount')
    for (const item of summary.review_candidates_by_name || [])
      add(item?.part_name, item?.count, 'reviewCount')
    return [...byName.values()].sort((a, b) =>
      a.partName.localeCompare(b.partName)
    )
  }

  // Backward compatibility for reports generated before V2.1.
  const instances = new Map()
  let fallbackIndex = 0

  for (const part of report?.damaged_parts || []) {
    const name = String(part?.part_name || '').trim()
    const regions = Array.isArray(part?.damage_regions)
      ? part.damage_regions
      : []
    if (!name || regions.length === 0) continue

    const suppliedId = String(part?.part_instance_id || '').trim()
    const instanceId = suppliedId || `fallback:${name}:${fallbackIndex++}`
    const confirmed =
      part?.damage_status === 'detected' ||
      regions.some((region) => region?.damage_status === 'detected')

    if (confirmed) instances.set(instanceId, { partName: name })
  }

  const counts = new Map()
  for (const instance of instances.values()) {
    const row = counts.get(instance.partName) || {
      partName: instance.partName,
      count: 0,
      confirmedCount: 0,
      reviewCount: 0
    }
    row.count += 1
    row.confirmedCount += 1
    counts.set(instance.partName, row)
  }

  return [...counts.values()].sort((a, b) =>
    a.partName.localeCompare(b.partName)
  )
}

function humanizeReason(reason) {
  return String(reason || '')
    .replaceAll('_', ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function getArtifactUrl(jobId, relativePath) {
  if (!jobId || !relativePath) return null
  const safePath = String(relativePath)
    .replaceAll('\\', '/')
    .split('/')
    .filter(Boolean)
    .map((part) => encodeURIComponent(part))
    .join('/')
  return `${API_BASE}/artifacts/${encodeURIComponent(jobId)}/${safePath}`
}

function formatTimestamp(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return 'Timestamp unavailable'
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds - minutes * 60
  return `${minutes}:${remainingSeconds.toFixed(2).padStart(5, '0')}`
}

function friendlyError(message) {
  if (!message) return 'Something went wrong while processing the video.'
  try {
    const parsed = JSON.parse(message)
    return parsed.detail || message
  } catch {
    return message
  }
}

function App() {
  const inputRef = useRef(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [jobId, setJobId] = useState('')
  const [jobStatus, setJobStatus] = useState('idle')
  const [report, setReport] = useState(null)
  const [evidenceFrames, setEvidenceFrames] = useState([])
  const [evidenceLoading, setEvidenceLoading] = useState(false)
  const [evidenceError, setEvidenceError] = useState('')
  const [selectedEvidence, setSelectedEvidence] = useState(null)
  const [error, setError] = useState('')
  const [apiHealth, setApiHealth] = useState({
    state: 'checking',
    message: 'Connecting to analysis service…'
  })

  useEffect(() => {
    let cancelled = false
    fetch(`${API_BASE}/health`)
      .then(async (response) => {
        const data = await response.json().catch(() => ({}))
        if (!response.ok)
          throw new Error(
            data.detail || 'The analysis service is not responding.'
          )
        if (cancelled) return
        const checkpointsReady =
          data.car_parts_checkpoint_found && data.damage_checkpoint_found
        setApiHealth({
          state: checkpointsReady ? 'ready' : 'warning',
          message: checkpointsReady
            ? 'Analysis service connected · both model checkpoints found'
            : 'API connected, but one or both model checkpoints are missing'
        })
      })
      .catch((err) => {
        if (!cancelled)
          setApiHealth({
            state: 'offline',
            message: err.message || 'Cannot connect to the analysis API'
          })
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!jobId) return undefined
    let cancelled = false
    let timer

    const poll = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/jobs/${jobId}`)
        const data = await response.json().catch(() => ({}))
        if (!response.ok)
          throw new Error(data.detail || 'Could not check analysis status.')
        if (cancelled) return

        if (data.status === 'failed') {
          setJobStatus('failed')
          setError(
            data.error ||
              'Video analysis failed. Check the backend terminal for details.'
          )
          return
        }
        if (data.status === 'completed') {
          setJobStatus('completed')
          const reportResponse = await fetch(
            `${API_BASE}/api/jobs/${jobId}/report`
          )
          const reportData = await reportResponse.json().catch(() => ({}))
          if (!reportResponse.ok)
            throw new Error(
              reportData.detail ||
                'Analysis finished, but the report could not be loaded.'
            )
          if (!cancelled) setReport(reportData)
          return
        }
        setJobStatus(data.status)
        timer = window.setTimeout(poll, 1800)
      } catch (err) {
        if (cancelled) return
        setError(err.message || 'Could not communicate with the analysis API.')
        setJobStatus('failed')
      }
    }

    poll()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [jobId])

  useEffect(() => {
    if (!jobId || jobStatus !== 'completed' || !report) return undefined
    let cancelled = false
    setEvidenceLoading(true)
    setEvidenceError('')

    fetch(`${API_BASE}/api/jobs/${jobId}/evidence-frames`)
      .then(async (response) => {
        const data = await response.json().catch(() => ({}))
        if (!response.ok)
          throw new Error(
            data.detail || 'Could not load evidence frame images.'
          )
        return data
      })
      .then((data) => {
        if (!cancelled)
          setEvidenceFrames(
            Array.isArray(data.evidence_frames) ? data.evidence_frames : []
          )
      })
      .catch((err) => {
        if (cancelled) return
        setEvidenceFrames([])
        setEvidenceError(err.message || 'Could not load evidence frame images.')
      })
      .finally(() => {
        if (!cancelled) setEvidenceLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [jobId, jobStatus, report])

  useEffect(() => {
    if (!selectedEvidence) return undefined
    const onKeyDown = (event) => {
      if (event.key === 'Escape') setSelectedEvidence(null)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [selectedEvidence])

  const partRows = useMemo(() => getPartRows(report), [report])
  const confirmedPartInstances = Number(
    report?.summary?.confirmed_part_instances ??
      partRows.reduce((sum, row) => sum + row.confirmedCount, 0)
  )
  // Review candidates are informational only and never contribute to the
  // confirmed-part count. They are grouped into partRows so the table can
  // surface review-only parts alongside confirmed ones.
  const reviewPartInstances = Number(
    report?.summary?.review_part_instances ??
      partRows.reduce((sum, row) => sum + row.reviewCount, 0)
  )
  const reviewCandidateTracks = Array.isArray(report?.review_candidates)
    ? report.review_candidates.length
    : 0
  const unassignedDamageTracks = Number(
    report?.summary?.unassigned_damage_regions ??
      report?.unassigned_damage_regions?.length ??
      0
  )
  const confirmedDamageTracks = Number(
    report?.summary?.confirmed_damage_regions ?? 0
  )

  const setFile = (file) => {
    setError('')
    setReport(null)
    setEvidenceFrames([])
    setEvidenceLoading(false)
    setEvidenceError('')
    setSelectedEvidence(null)
    setJobId('')
    setJobStatus('idle')
    if (!file) {
      setSelectedFile(null)
      return
    }
    const extension = `.${(file.name.split('.').pop() || '').toLowerCase()}`
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      setSelectedFile(null)
      setError(
        'Choose a supported video file: MP4, MOV, AVI, MKV, WEBM or M4V.'
      )
      return
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setSelectedFile(null)
      setError('This video is larger than the 500 MB upload limit.')
      return
    }
    if (file.size === 0) {
      setSelectedFile(null)
      setError('The selected file is empty. Please choose another video.')
      return
    }
    setSelectedFile(file)
  }

  const handleUpload = async (event) => {
    event.preventDefault()
    if (
      !selectedFile ||
      uploading ||
      ['queued', 'processing'].includes(jobStatus)
    )
      return

    setUploading(true)
    setError('')
    setReport(null)
    setEvidenceFrames([])
    setEvidenceLoading(false)
    setEvidenceError('')
    setSelectedEvidence(null)
    setJobId('')
    setJobStatus('idle')
    try {
      const formData = new FormData()
      formData.append('video', selectedFile)
      const response = await fetch(`${API_BASE}/api/jobs`, {
        method: 'POST',
        body: formData
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(data.detail || 'Video upload failed.')
      setJobId(data.job_id)
      setJobStatus(data.status || 'queued')
    } catch (err) {
      setError(
        err.message ||
          'Could not upload the video. Make sure the backend is running.'
      )
    } finally {
      setUploading(false)
    }
  }

  const resetAnalysis = () => {
    setSelectedFile(null)
    setJobId('')
    setJobStatus('idle')
    setReport(null)
    setEvidenceFrames([])
    setEvidenceLoading(false)
    setEvidenceError('')
    setSelectedEvidence(null)
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  const busy = uploading || ['queued', 'processing'].includes(jobStatus)

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Car Damage Analyzer home">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" fill="none">
              <path
                d="M5 19.5 7.7 12a3 3 0 0 1 2.8-2h11a3 3 0 0 1 2.8 2l2.7 7.5v5H5v-5Z"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinejoin="round"
              />
              <path
                d="m9 12.2 1.4 5h11.2l1.4-5M5.5 20h21M9 24.5v2m14-2v2"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
              />
              <circle cx="10" cy="20.5" r="1" fill="currentColor" />
              <circle cx="22" cy="20.5" r="1" fill="currentColor" />
            </svg>
          </span>
          <span className="brand-copy">
            <strong>CAR SCOPE</strong>
            <small>VIDEO DAMAGE ANALYSIS</small>
          </span>
        </a>
        <div
          className={`service-status ${apiHealth.state}`}
          title={apiHealth.message}
        >
          <span className="status-dot" />
          <span>
            {apiHealth.state === 'ready'
              ? 'System ready'
              : apiHealth.state === 'checking'
                ? 'Connecting'
                : apiHealth.state === 'warning'
                  ? 'Model check needed'
                  : 'API unavailable'}
          </span>
        </div>
      </header>

      <main id="top" className="main-content">
        <section className="intro-section">
          <div className="eyebrow">
            <span className="eyebrow-line" /> AI-POWERED VEHICLE INSPECTION
          </div>
          <h1>
            See the damage.
            <br />
            <span>Know the parts.</span>
          </h1>
          <p className="intro-copy">
            Upload a walk-around video of a vehicle. The analysis pipeline scans
            frames, identifies affected car parts, and summarizes the results
            for you.
          </p>
        </section>

        <section className="workspace-grid">
          <form className="upload-card panel" onSubmit={handleUpload}>
            <div className="section-heading">
              <div>
                <span className="step-label">STEP 01</span>
                <h2>Upload video</h2>
              </div>
              <span className="file-type-tag">MP4 · MOV · AVI</span>
            </div>

            <div
              className={`dropzone ${dragging ? 'dragging' : ''} ${selectedFile ? 'has-file' : ''}`}
              onDragOver={(event) => {
                event.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault()
                setDragging(false)
                setFile(event.dataTransfer.files?.[0])
              }}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  inputRef.current?.click()
                }
              }}
              onClick={(event) => {
                if (event.target !== inputRef.current) inputRef.current?.click()
              }}
              aria-label="Choose a video file or drop it here"
            >
              <input
                ref={inputRef}
                className="hidden-input"
                type="file"
                accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,video/x-m4v,.mp4,.mov,.avi,.mkv,.webm,.m4v"
                onChange={(event) => setFile(event.target.files?.[0])}
              />
              <div className="upload-icon-wrap">
                {selectedFile ? (
                  <svg viewBox="0 0 24 24" fill="none">
                    <path
                      d="M5 12.5 9.2 17 19 7"
                      stroke="currentColor"
                      strokeWidth="1.8"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                ) : (
                  <svg viewBox="0 0 24 24" fill="none">
                    <path
                      d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4 16.5v2A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5v-2"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                )}
              </div>
              {selectedFile ? (
                <>
                  <strong className="file-name">{selectedFile.name}</strong>
                  <span className="drop-hint">
                    {formatBytes(selectedFile.size)} · Ready to analyze
                  </span>
                </>
              ) : (
                <>
                  <strong>Drop your video here</strong>
                  <span className="drop-hint">
                    or <span className="browse-link">browse files</span>
                  </span>
                  <span className="drop-footnote">
                    Maximum file size: 500 MB
                  </span>
                </>
              )}
            </div>

            {error && (
              <div className="error-message" role="alert">
                <span className="error-symbol">!</span>
                <span>{friendlyError(error)}</span>
              </div>
            )}

            <button
              className="primary-button"
              type="submit"
              disabled={!selectedFile || busy || apiHealth.state === 'offline'}
            >
              {uploading ? (
                <>
                  <span className="button-spinner" /> Uploading video…
                </>
              ) : busy ? (
                <>
                  <span className="button-spinner" /> Analyzing video…
                </>
              ) : (
                <>
                  Analyze video <span aria-hidden="true">↗</span>
                </>
              )}
            </button>
            <p className="privacy-note">
              <span className="lock-icon" aria-hidden="true">
                ⌑
              </span>{' '}
              Your video is processed by the connected analysis server.
            </p>
          </form>

          <section
            className="results-card panel"
            aria-labelledby="results-title"
          >
            <div className="section-heading results-heading">
              <div>
                <span className="step-label">STEP 02</span>
                <h2 id="results-title">Detection summary</h2>
              </div>
              {report && (
                <span className="complete-tag">
                  <span /> Complete
                </span>
              )}
            </div>

            {!report && !busy && jobStatus !== 'failed' && (
              <div className="empty-results">
                <div className="empty-illustration" aria-hidden="true">
                  <svg viewBox="0 0 150 112" fill="none">
                    <rect
                      x="20"
                      y="15"
                      width="110"
                      height="78"
                      rx="8"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeDasharray="4 5"
                    />
                    <path
                      d="m35 72 20-21 15 14 13-16 31 30"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <circle
                      cx="100"
                      cy="34"
                      r="7"
                      stroke="currentColor"
                      strokeWidth="1.5"
                    />
                    <path
                      d="M54 100h42"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                    />
                  </svg>
                </div>
                <h3>Your results will appear here</h3>
                <p>
                  Upload a vehicle video to see the affected part names and
                  counts.
                </p>
                <div className="table-preview" aria-hidden="true">
                  <div>
                    <span>Damaged part name</span>
                    <span>Count</span>
                  </div>
                  <div>
                    <i />
                    <i />
                  </div>
                  <div>
                    <i />
                    <i />
                  </div>
                </div>
              </div>
            )}

            {busy && (
              <div className="processing-state" aria-live="polite">
                <div className="processing-orbit">
                  <span />
                  <span />
                  <span />
                </div>
                <h3>
                  {jobStatus === 'queued'
                    ? 'Your video is in the queue'
                    : 'Analyzing your video'}
                </h3>
                <p>
                  {jobStatus === 'queued'
                    ? 'The analysis service will start processing shortly.'
                    : 'Both vision models are reviewing the sampled frames. Longer videos may take more time.'}
                </p>
                <div className="progress-track">
                  <span className="progress-indeterminate" />
                </div>
                <span className="processing-caption">
                  PLEASE KEEP THIS PAGE OPEN
                </span>
              </div>
            )}

            {report && (
              <div className="results-content">
                <div className="metric-row">
                  <div className="metric-icon" aria-hidden="true">
                    <svg viewBox="0 0 24 24" fill="none">
                      <path
                        d="M4 7.5 12 3l8 4.5v9L12 21l-8-4.5v-9Z"
                        stroke="currentColor"
                        strokeWidth="1.6"
                        strokeLinejoin="round"
                      />
                      <path
                        d="m4.5 7.8 7.5 4.4 7.5-4.4M12 12.5V21"
                        stroke="currentColor"
                        strokeWidth="1.6"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </div>
                  <div className="metric-copy">
                    <span>Affected car parts</span>
                    <strong>{partRows.length}</strong>
                  </div>
                  <span className="metric-subtext">
                    {confirmedPartInstances} confirmed · {reviewPartInstances}{' '}
                    pending review
                  </span>
                </div>
                <div className="table-topline">
                  <div>
                    <span className="table-title">Damaged parts</span>
                    <span className="table-subtitle">
                      Grouped by part name · confirmed parts and parts pending
                      manual review are shown together
                    </span>
                  </div>
                  <span className="row-count">
                    {partRows.length} {partRows.length === 1 ? 'row' : 'rows'}
                  </span>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th scope="col">Part name</th>
                        <th scope="col" className="count-column">
                          Part instances
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {partRows.length ? (
                        partRows.map((row) => (
                          <tr key={row.partName}>
                            <td>
                              <span className="part-cell">
                                <span className="part-dot" />
                                {row.partName.replaceAll('_', ' ')}
                              </span>
                              {row.reviewCount > 0 && (
                                <span
                                  title={`${row.reviewCount} part instance(s) need review`}
                                  style={{
                                    display: 'inline-block',
                                    margin: '5px 0 0 18px',
                                    padding: '3px 7px',
                                    borderRadius: '999px',
                                    background: '#fff4df',
                                    color: '#8a5a00',
                                    fontSize: '11px',
                                    lineHeight: 1.4,
                                    fontWeight: 600
                                  }}
                                >
                                  {row.confirmedCount > 0
                                    ? 'Some need review'
                                    : 'Needs review'}
                                </span>
                              )}
                            </td>
                            <td className="count-column">
                              <span className="count-pill">{row.count}</span>
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan="2" className="no-damage-cell">
                            <span className="no-damage-icon">✓</span>
                            <strong>
                              No damaged car parts detected in this video
                            </strong>
                            <span>
                              The damage model found {reviewCandidateTracks}{' '}
                              track(s), but none could be associated with a car
                              part.
                            </span>
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
                <div className="result-footer">
                  <span>
                    <span className="footer-dot" /> Report generated
                  </span>
                  <span>
                    {report.video?.processed_frames ?? '—'} frames analyzed
                  </span>
                </div>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={resetAnalysis}
                >
                  Analyze another video <span aria-hidden="true">↗</span>
                </button>
                <div
                  className="v21-report-stats"
                  aria-label="V2.1 damage association summary"
                >
                  <div>
                    <strong>{confirmedDamageTracks}</strong>
                    <span>confirmed damage tracks</span>
                  </div>
                  <div>
                    <strong>{reviewCandidateTracks}</strong>
                    <span>tracks with part candidates</span>
                  </div>
                  <div>
                    <strong>{unassignedDamageTracks}</strong>
                    <span>unassigned tracks</span>
                  </div>
                </div>
                {(reviewPartInstances > 0 || unassignedDamageTracks > 0) && (
                  <p className="review-note">
                    V2.1 found {reviewPartInstances} named part instance(s) that
                    require manual review and {unassignedDamageTracks}{' '}
                    unassigned damage track(s). The {reviewCandidateTracks}{' '}
                    review entries are damage tracks, not unique part counts.
                    Evidence images show model predictions; they do not by
                    themselves prove physical damage or a confirmed part
                    association.
                  </p>
                )}
                {reviewCandidateTracks > 0 && (
                  <details className="v21-review-details">
                    <summary>
                      Inspect {reviewCandidateTracks} damage tracks requiring
                      association review
                    </summary>
                    <div className="v21-review-list">
                      {report.review_candidates.map((candidate, index) => {
                        const region =
                          candidate?.damage_region || candidate || {}
                        const candidateNames = Array.isArray(
                          candidate?.candidate_part_names
                        )
                          ? candidate.candidate_part_names
                          : Array.isArray(region?.candidate_part_names)
                            ? region.candidate_part_names
                            : []
                        const primaryName =
                          candidate?.candidate_part_name ||
                          candidateNames[0] ||
                          'No reliable part candidate'
                        const reasons = Array.isArray(candidate?.review_reasons)
                          ? candidate.review_reasons
                          : Array.isArray(region?.review_reasons)
                            ? region.review_reasons
                            : []
                        const representativePath =
                          region?.representative_frame_path
                        const evidenceUrl = getArtifactUrl(
                          jobId,
                          representativePath
                        )
                        const score = Number(region?.damage_confidence)
                        const consistency = Number(
                          candidate?.association_consistency ??
                            region?.association_consistency
                        )
                        const frameNumber = region?.representative_frame
                        const damageId =
                          candidate?.damage_id ||
                          region?.damage_id ||
                          `review-${index + 1}`

                        return (
                          <article className="v21-review-item" key={damageId}>
                            <div className="v21-review-item-heading">
                              <div>
                                <strong>
                                  {String(primaryName).replaceAll('_', ' ')}
                                </strong>
                                <span>{damageId}</span>
                              </div>
                              <span className="v21-review-badge">
                                Manual review
                              </span>
                            </div>
                            <div className="v21-review-meta">
                              <span>
                                Raw model score:{' '}
                                {Number.isFinite(score)
                                  ? score.toFixed(3)
                                  : '—'}
                              </span>
                              <span>
                                Association consistency:{' '}
                                {Number.isFinite(consistency)
                                  ? `${(consistency * 100).toFixed(1)}%`
                                  : '—'}
                              </span>
                              <span>
                                Representative frame:{' '}
                                {frameNumber != null ? frameNumber : '—'}
                              </span>
                              {evidenceUrl && (
                                <a
                                  href={evidenceUrl}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  Open evidence frame ↗
                                </a>
                              )}
                            </div>
                            {candidateNames.length > 1 && (
                              <p className="v21-review-alternatives">
                                Other candidates:{' '}
                                {candidateNames
                                  .filter((name) => name !== primaryName)
                                  .map((name) =>
                                    String(name).replaceAll('_', ' ')
                                  )
                                  .join(', ')}
                              </p>
                            )}
                            {reasons.length > 0 && (
                              <p className="v21-review-reasons">
                                {reasons
                                  .slice(0, 4)
                                  .map(humanizeReason)
                                  .join(' · ')}
                              </p>
                            )}
                          </article>
                        )
                      })}
                    </div>
                  </details>
                )}
              </div>
            )}

            {jobStatus === 'failed' && !report && (
              <div className="failed-state">
                <div className="failed-icon">!</div>
                <h3>Analysis could not be completed</h3>
                <p>{friendlyError(error)}</p>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={resetAnalysis}
                >
                  Try another video
                </button>
              </div>
            )}
          </section>
        </section>

        {report && (
          <section
            className="evidence-card panel"
            aria-labelledby="evidence-title"
          >
            <div className="evidence-heading">
              <div>
                <span className="step-label">STEP 03</span>
                <h2 id="evidence-title">Evidence frames</h2>
                <p className="evidence-description">
                  Annotated frames containing model-predicted damage regions.
                  Review these predictions before treating them as confirmed
                  damage.
                </p>
              </div>
              <div className="evidence-heading-actions">
                <span className="row-count">
                  {evidenceFrames.length}{' '}
                  {evidenceFrames.length === 1 ? 'image' : 'images'}
                </span>
                {evidenceFrames.length > 0 && (
                  <a
                    className="download-evidence-button"
                    href={`${API_BASE}/api/jobs/${jobId}/evidence-frames/download`}
                    download
                    title="Download all evidence images in one ZIP file"
                  >
                    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                      <path
                        d="M12 3v11m0 0 4-4m-4 4-4-4M4 17v3h16v-3"
                        stroke="currentColor"
                        strokeWidth="1.7"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                    Download all images
                  </a>
                )}
              </div>
            </div>

            {evidenceLoading && (
              <div className="evidence-state">
                <span className="button-spinner evidence-spinner" /> Loading
                evidence images…
              </div>
            )}
            {!evidenceLoading && evidenceError && (
              <div className="evidence-state evidence-error" role="alert">
                {evidenceError} Refresh the page or check that the backend is
                running.
              </div>
            )}
            {!evidenceLoading &&
              !evidenceError &&
              evidenceFrames.length === 0 && (
                <div className="evidence-empty">
                  <div className="evidence-empty-icon" aria-hidden="true">
                    ▧
                  </div>
                  <strong>No evidence images were generated</strong>
                  <span>
                    The pipeline only saves evidence frames when it detects a
                    damage region.
                  </span>
                </div>
              )}
            {!evidenceLoading &&
              !evidenceError &&
              evidenceFrames.length > 0 && (
                <div className="evidence-grid">
                  {evidenceFrames.map((frame) => {
                    const imageUrl = `${API_BASE}${frame.url}`
                    const timestamp = formatTimestamp(frame.timestamp_seconds)
                    return (
                      <button
                        className="evidence-thumb"
                        type="button"
                        key={frame.filename}
                        onClick={() => setSelectedEvidence(frame)}
                        aria-label={`View evidence image for frame ${frame.frame_number ?? frame.filename}`}
                      >
                        <div className="evidence-image-wrap">
                          <img
                            src={imageUrl}
                            alt={`Annotated car damage evidence at ${timestamp}`}
                            loading="lazy"
                          />
                          <span className="evidence-view-chip">
                            View image ↗
                          </span>
                        </div>
                        <span className="evidence-thumb-meta">
                          <strong>
                            {frame.frame_number != null
                              ? `Frame ${frame.frame_number}`
                              : frame.filename}
                          </strong>
                          <span>{timestamp}</span>
                        </span>
                      </button>
                    )
                  })}
                </div>
              )}
          </section>
        )}

        {selectedEvidence && (
          <div
            className="evidence-modal-backdrop"
            role="presentation"
            onMouseDown={(event) => {
              if (event.target === event.currentTarget)
                setSelectedEvidence(null)
            }}
          >
            <section
              className="evidence-modal"
              role="dialog"
              aria-modal="true"
              aria-labelledby="evidence-modal-title"
            >
              <div className="evidence-modal-header">
                <div>
                  <span className="step-label">ANNOTATED FRAME</span>
                  <h2 id="evidence-modal-title">
                    {selectedEvidence.frame_number != null
                      ? `Frame ${selectedEvidence.frame_number}`
                      : selectedEvidence.filename}
                  </h2>
                  <span className="evidence-modal-time">
                    {formatTimestamp(selectedEvidence.timestamp_seconds)}
                  </span>
                </div>
                <button
                  className="modal-close-button"
                  type="button"
                  onClick={() => setSelectedEvidence(null)}
                  aria-label="Close image preview"
                >
                  ×
                </button>
              </div>
              <img
                className="evidence-modal-image"
                src={`${API_BASE}${selectedEvidence.url}`}
                alt={`Annotated evidence image ${selectedEvidence.filename}`}
              />
              <div className="evidence-modal-footer">
                <span>{selectedEvidence.filename}</span>
                <a
                  href={`${API_BASE}${selectedEvidence.url}`}
                  download={selectedEvidence.filename}
                >
                  Download this image
                </a>
              </div>
            </section>
          </div>
        )}

        <footer className="page-footer">
          <span>
            CAR SCOPE <span className="footer-divider">/</span> VIDEO INSPECTION
          </span>
          <span>AI-assisted results · Review important decisions manually</span>
        </footer>
      </main>
    </div>
  )
}

export default App
