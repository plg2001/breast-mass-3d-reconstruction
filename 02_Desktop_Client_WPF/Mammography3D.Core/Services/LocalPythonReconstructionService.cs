using System;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Mammography3D.Core.Models;

namespace Mammography3D.Core.Services
{
    /// <summary>
    /// Executes the Python 3D reconstruction pipeline as a background process.
    /// Streams real-time progress and parses validation metrics upon completion.
    /// </summary>
    public class LocalPythonReconstructionService : IReconstructionService
    {
        public async Task<ReconstructionResult> ReconstructAsync(
            ReconstructionConfig config,
            IProgress<ReconstructionProgress>? progress = null,
            CancellationToken cancellationToken = default)
        {
            var sw = Stopwatch.StartNew();

            // 1. Validation
            var validationErrors = config.Validate();
            if (validationErrors.Count > 0)
            {
                return ReconstructionResult.Fail(string.Join("\n", validationErrors));
            }

            // 2. Resolve Python and Script paths
            string pythonExe = ResolvePythonPath(config.PythonExecutable);
            string scriptPath = ResolveScriptPath(config.Algorithm, config.ScriptPath);

            if (!File.Exists(pythonExe) && pythonExe != "python")
            {
                return ReconstructionResult.Fail($"Interprete Python non trovato. Percorso cercato: {pythonExe}");
            }
            if (!File.Exists(scriptPath))
            {
                return ReconstructionResult.Fail($"Script di ricostruzione non trovato in: {scriptPath}");
            }

            string outputDir = string.IsNullOrWhiteSpace(config.OutputDir)
                ? Path.Combine(Path.GetDirectoryName(scriptPath) ?? ".", "output")
                : Path.GetFullPath(config.OutputDir);

            Directory.CreateDirectory(outputDir);

            // 3. Build CLI arguments based on selected algorithm
            string arguments;
            if (config.Algorithm == ReconstructionAlgorithm.SMRD_2015)
            {
                arguments = $"\"{scriptPath}\"" +
                    $" --roi-cc \"{Path.GetFullPath(config.RoiCcPath)}\"" +
                    $" --roi-mlo \"{Path.GetFullPath(config.RoiMloPath)}\"" +
                    $" --output-dir \"{outputDir}\"" +
                    $" --beta {config.MloAngleDeg.ToString(System.Globalization.CultureInfo.InvariantCulture)}" +
                    $" --grid-resolution {config.GridResolution}" +
                    $" --voxel-size-mm {config.VoxelSizeMm.ToString(System.Globalization.CultureInfo.InvariantCulture)}";

                if (!string.IsNullOrWhiteSpace(config.RCcPath) && File.Exists(config.RCcPath))
                    arguments += $" --r-cc \"{Path.GetFullPath(config.RCcPath)}\"";
                if (!string.IsNullOrWhiteSpace(config.RMloPath) && File.Exists(config.RMloPath))
                    arguments += $" --r-mlo \"{Path.GetFullPath(config.RMloPath)}\"";
            }
            else
            {
                arguments = $"\"{scriptPath}\"" +
                    $" --r-cc \"{Path.GetFullPath(config.RCcPath)}\"" +
                    $" --r-mlo \"{Path.GetFullPath(config.RMloPath)}\"" +
                    $" --roi-cc \"{Path.GetFullPath(config.RoiCcPath)}\"" +
                    $" --roi-mlo \"{Path.GetFullPath(config.RoiMloPath)}\"" +
                    $" --output-dir \"{outputDir}\"" +
                    $" --mlo-angle {config.MloAngleDeg.ToString(System.Globalization.CultureInfo.InvariantCulture)}" +
                    $" --grid-resolution {config.GridResolution}" +
                    $" --voxel-size-mm {config.VoxelSizeMm.ToString(System.Globalization.CultureInfo.InvariantCulture)}";
            }

            progress?.Report(new ReconstructionProgress(
                ReconstructionPhase.Starting,
                "Inizializzazione ambiente e parametri...",
                2.0f,
                $"Avvio Python: {pythonExe} con script {Path.GetFileName(scriptPath)}"
            ));

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                WorkingDirectory = Path.GetDirectoryName(scriptPath) ?? Directory.GetCurrentDirectory()
            };

            using var process = new Process { StartInfo = psi, EnableRaisingEvents = true };
            var tcs = new TaskCompletionSource<int>();

            float currentPercent = 2.0f;
            ReconstructionPhase currentPhase = ReconstructionPhase.Starting;

            void ProcessOutputLine(string line, bool isStderr)
            {
                if (string.IsNullOrWhiteSpace(line)) return;

                // SMRD Stages
                if (line.Contains("[SMRD] Inizializzazione"))
                {
                    currentPhase = ReconstructionPhase.Starting;
                    currentPercent = Math.Max(currentPercent, 10.0f);
                }
                else if (line.Contains("[SMRD] Caricamento maschere"))
                {
                    currentPhase = ReconstructionPhase.SegmentationCC;
                    currentPercent = Math.Max(currentPercent, 25.0f);
                }
                else if (line.Contains("[SMRD] Fitting Armoniche Sferiche"))
                {
                    currentPhase = ReconstructionPhase.VisualHull;
                    currentPercent = Math.Max(currentPercent, 50.0f);
                }
                else if (line.Contains("[SMRD] Generazione superficie"))
                {
                    currentPhase = ReconstructionPhase.MeshExtraction;
                    currentPercent = Math.Max(currentPercent, 80.0f);
                }
                else if (line.Contains("[SMRD] Esportazione mesh"))
                {
                    currentPhase = ReconstructionPhase.ExportingGLB;
                    currentPercent = Math.Max(currentPercent, 90.0f);
                }
                else if (line.Contains("[SMRD] Generazione rendering"))
                {
                    currentPercent = Math.Max(currentPercent, 95.0f);
                }
                // Ling Shao Stages
                else if (line.Contains("[CC] Starting automated mass segmentation"))
                {
                    currentPhase = ReconstructionPhase.SegmentationCC;
                    currentPercent = Math.Max(currentPercent, 8.0f);
                }
                else if (line.Contains("[CC] Region growing completed"))
                {
                    currentPercent = Math.Max(currentPercent, 18.0f);
                }
                else if (line.Contains("[MLO] Starting automated mass segmentation"))
                {
                    currentPhase = ReconstructionPhase.SegmentationMLO;
                    currentPercent = Math.Max(currentPercent, 22.0f);
                }
                else if (line.Contains("[MLO] Region growing completed"))
                {
                    currentPercent = Math.Max(currentPercent, 32.0f);
                }
                else if (line.Contains("STEP 1/3") || line.Contains("Shape from Silhouettes"))
                {
                    currentPhase = ReconstructionPhase.VisualHull;
                    currentPercent = Math.Max(currentPercent, 38.0f);
                }
                else if (line.Contains("STEP 2/3") || line.Contains("Algebraic Reconstruction"))
                {
                    currentPhase = ReconstructionPhase.ART;
                    currentPercent = Math.Max(currentPercent, 45.0f);
                }
                else if (line.Contains("ART Iteration"))
                {
                    var match = Regex.Match(line, @"ART Iteration\s+(\d+)");
                    if (match.Success && int.TryParse(match.Groups[1].Value, out int iter))
                    {
                        currentPercent = Math.Min(62.0f, 45.0f + iter * 3.0f);
                    }
                }
                else if (line.Contains("STEP 3/3") || line.Contains("Partial Volume Estimation"))
                {
                    currentPhase = ReconstructionPhase.PartialVolumeEstimation;
                    currentPercent = Math.Max(currentPercent, 65.0f);
                }
                else if (line.Contains("PVE Iteration"))
                {
                    var match = Regex.Match(line, @"PVE Iteration\s+(\d+)");
                    if (match.Success && int.TryParse(match.Groups[1].Value, out int iter))
                    {
                        currentPercent = Math.Min(82.0f, 65.0f + iter * 2.5f);
                    }
                }
                else if (line.Contains("Extracting 3D surface mesh"))
                {
                    currentPhase = ReconstructionPhase.MeshExtraction;
                    currentPercent = Math.Max(currentPercent, 85.0f);
                }
                else if (line.Contains("Exporting 3D Meshes"))
                {
                    currentPhase = ReconstructionPhase.ExportingGLB;
                    currentPercent = Math.Max(currentPercent, 90.0f);
                }
                else if (line.Contains("Generating Diagnostic Figures"))
                {
                    currentPercent = Math.Max(currentPercent, 95.0f);
                }
                else if (line.Contains("PIPELINE COMPLETED SUCCESSFULLY"))
                {
                    currentPhase = ReconstructionPhase.Done;
                    currentPercent = 100.0f;
                }

                string stageDesc = GetPhaseDescription(currentPhase);
                progress?.Report(new ReconstructionProgress(currentPhase, stageDesc, currentPercent, line, isStderr));
            }

            process.OutputDataReceived += (_, e) => { if (e.Data != null) ProcessOutputLine(e.Data, false); };
            process.ErrorDataReceived += (_, e) => { if (e.Data != null) ProcessOutputLine(e.Data, true); };
            process.Exited += (_, _) => tcs.TrySetResult(process.ExitCode);

            try
            {
                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                using (cancellationToken.Register(() =>
                {
                    try
                    {
                        if (!process.HasExited)
                        {
                            process.Kill(true);
                            tcs.TrySetCanceled();
                        }
                    }
                    catch { }
                }))
                {
                    int exitCode = await tcs.Task.ConfigureAwait(false);
                    sw.Stop();

                    if (exitCode != 0)
                    {
                        return ReconstructionResult.Fail($"Processo terminato con codice di errore: {exitCode}", sw.Elapsed.TotalSeconds);
                    }

                    // 4. Parse results
                    return ParseReport(outputDir, sw.Elapsed.TotalSeconds);
                }
            }
            catch (OperationCanceledException)
            {
                progress?.Report(new ReconstructionProgress(
                    ReconstructionPhase.Cancelled,
                    "Operazione annullata dall'utente.",
                    0.0f,
                    "Ricostruzione interrotta."
                ));
                return ReconstructionResult.Fail("Operazione annullata.", sw.Elapsed.TotalSeconds);
            }
            catch (Exception ex)
            {
                return ReconstructionResult.Fail($"Errore di esecuzione: {ex.Message}", sw.Elapsed.TotalSeconds);
            }
        }

        private static string GetPhaseDescription(ReconstructionPhase phase) => phase switch
        {
            ReconstructionPhase.Starting => "Inizializzazione...",
            ReconstructionPhase.SegmentationCC => "Segmentazione 2D vista CC (Eqs. 1-6)...",
            ReconstructionPhase.SegmentationMLO => "Segmentazione 2D vista MLO (Eqs. 1-6)...",
            ReconstructionPhase.VisualHull => "Inizializzazione Visual Hull 3D baricentrico...",
            ReconstructionPhase.ART => "Raffinamento bordi Algebraic Reconstruction (ART)...",
            ReconstructionPhase.PartialVolumeEstimation => "Stima sub-voxel Partial Volume Estimation (PVE)...",
            ReconstructionPhase.MeshExtraction => "Estrazione isosuperficie e smoothing Taubin...",
            ReconstructionPhase.ExportingGLB => "Esportazione modelli 3D (.glb, .blend, .obj)...",
            ReconstructionPhase.Done => "Ricostruzione completata con successo!",
            ReconstructionPhase.Cancelled => "Annullato",
            _ => "Elaborazione in corso..."
        };

        private static string ResolvePythonPath(string? customPython)
        {
            if (!string.IsNullOrWhiteSpace(customPython) && File.Exists(customPython))
                return customPython;

            // 1. Check known system-wide installation locations (e.g. C:\Python314\python.exe)
            var directLocations = new[]
            {
                @"C:\Python311\python.exe",
                @"C:\Python312\python.exe",
                @"C:\Python310\python.exe",
                @"C:\Python314\python.exe",
                @"C:\Python310\python.exe",
                @"C:\Python39\python.exe",
                @"C:\Program Files\Python314\python.exe",
                @"C:\Program Files\Python313\python.exe",
                @"C:\Program Files\Python312\python.exe",
            };
            foreach (var loc in directLocations)
            {
                if (File.Exists(loc)) return loc;
            }

            // 2. Search PATH environment variable
            string? pathEnv = Environment.GetEnvironmentVariable("PATH");
            if (!string.IsNullOrEmpty(pathEnv))
            {
                string[] paths = pathEnv.Split(Path.PathSeparator);
                foreach (string dir in paths)
                {
                    try
                    {
                        string trimmed = dir.Trim('\"', ' ');
                        if (string.IsNullOrEmpty(trimmed) || trimmed.Contains("WindowsApps", StringComparison.OrdinalIgnoreCase))
                            continue; // skip WindowsApps redirector alias

                        string candidate = Path.Combine(trimmed, "python.exe");
                        if (File.Exists(candidate))
                            return candidate;
                    }
                    catch { }
                }
            }

            // 3. Check AppData Local Programs
            try
            {
                string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                string programsPython = Path.Combine(localAppData, "Programs", "Python");
                if (Directory.Exists(programsPython))
                {
                    foreach (string dir in Directory.GetDirectories(programsPython))
                    {
                        string candidate = Path.Combine(dir, "python.exe");
                        if (File.Exists(candidate)) return candidate;
                    }
                }
            }
            catch { }

            // 4. Check Blender portable embedded Python
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            var blenderPyCandidates = new[]
            {
                @"C:\Users\Pierluca\OneDrive\Documenti\Blender\5.2\python\bin\python.exe",
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "OneDrive", "Documenti", "Blender", "5.2", "python", "bin", "python.exe"),
                Path.Combine(baseDir, "..", "Blender", "5.2", "python", "bin", "python.exe"),
                Path.Combine(baseDir, "..", "..", "..", "..", "Blender", "5.2", "python", "bin", "python.exe"),
                @"C:\Users\Pierluca\Desktop\Tesi\Blender\5.2\python\bin\python.exe"
            };
            foreach (var bp in blenderPyCandidates)
            {
                if (File.Exists(bp)) return Path.GetFullPath(bp);
            }

            return "python";
        }

        private static string FindRepoRoot()
        {
            string current = AppDomain.CurrentDomain.BaseDirectory;
            for (int i = 0; i < 6; i++)
            {
                if (Directory.Exists(Path.Combine(current, "01_Baseline_Lesion_LingShao")) ||
                    Directory.Exists(Path.Combine(current, "01_Ricostruzione_Lesione_3D")))
                {
                    return current;
                }
                var parent = Directory.GetParent(current);
                if (parent == null) break;
                current = parent.FullName;
            }
            return @"c:\Users\Pierluca\Desktop\breast-mass-3d-reconstruction";
        }

        private static string ResolveScriptPath(ReconstructionAlgorithm algorithm, string? customScript)
        {
            if (!string.IsNullOrWhiteSpace(customScript) && File.Exists(customScript))
                return customScript;

            string repoRoot = FindRepoRoot();
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;

            if (algorithm == ReconstructionAlgorithm.SMRD_2015)
            {
                string smrdCandidate = Path.Combine(repoRoot, "04_Spicular_Lesion_SMRD_Module", "smrd_lesion_reconstruction.py");
                if (File.Exists(smrdCandidate))
                    return smrdCandidate;

                var smrdCandidates = new[]
                {
                    Path.Combine(baseDir, "..", "04_Spicular_Lesion_SMRD_Module", "smrd_lesion_reconstruction.py"),
                    Path.Combine(baseDir, "..", "..", "04_Spicular_Lesion_SMRD_Module", "smrd_lesion_reconstruction.py"),
                    Path.Combine(baseDir, "04_Spicular_Lesion_SMRD_Module", "smrd_lesion_reconstruction.py"),
                    @"c:\Users\Pierluca\Desktop\breast-mass-3d-reconstruction\04_Spicular_Lesion_SMRD_Module\smrd_lesion_reconstruction.py",
                    @"c:\Users\Pierluca\Desktop\Tesi\04_Spicular_Lesion_SMRD_Module\smrd_lesion_reconstruction.py"
                };

                foreach (var c in smrdCandidates)
                {
                    try { if (File.Exists(c)) return Path.GetFullPath(c); } catch { }
                }
            }

            string candidate = Path.Combine(repoRoot, "01_Baseline_Lesion_LingShao", "lesion_pvc_reconstruction.py");
            if (File.Exists(candidate))
                return candidate;

            var candidates = new[]
            {
                Path.Combine(baseDir, "..", "01_Baseline_Lesion_LingShao", "lesion_pvc_reconstruction.py"),
                Path.Combine(baseDir, "..", "..", "01_Baseline_Lesion_LingShao", "lesion_pvc_reconstruction.py"),
                Path.Combine(baseDir, "01_Baseline_Lesion_LingShao", "lesion_pvc_reconstruction.py"),
                @"c:\Users\Pierluca\Desktop\breast-mass-3d-reconstruction\01_Baseline_Lesion_LingShao\lesion_pvc_reconstruction.py",
                @"c:\Users\Pierluca\Desktop\Tesi\01_Baseline_Lesion_LingShao\lesion_pvc_reconstruction.py",
                // Fallback precedenti
                Path.Combine(baseDir, "..", "01_Ricostruzione_Lesione_3D", "lesion_pvc_reconstruction.py"),
                @"c:\Users\Pierluca\Desktop\breast-mass-3d-reconstruction\01_Ricostruzione_Lesione_3D\lesion_pvc_reconstruction.py"
            };

            foreach (var c in candidates)
            {
                try
                {
                    if (File.Exists(c)) return Path.GetFullPath(c);
                }
                catch { }
            }

            return @"c:\Users\Pierluca\Desktop\breast-mass-3d-reconstruction\01_Baseline_Lesion_LingShao\lesion_pvc_reconstruction.py";
        }

        private static ReconstructionResult ParseReport(string outputDir, double executionTime)
        {
            var res = new ReconstructionResult
            {
                Success = true,
                ExecutionTimeSeconds = executionTime,
                OutputDirectory = outputDir
            };

            string meshesDir = Path.Combine(outputDir, "meshes");
            string figuresDir = Path.Combine(outputDir, "figures");

            res.GlbLesionPath = Path.Combine(meshesDir, "lesion_reconstructed.glb");
            if (!File.Exists(res.GlbLesionPath) && Directory.Exists(outputDir))
            {
                var glbs = Directory.GetFiles(outputDir, "*.glb");
                if (glbs.Length > 0) res.GlbLesionPath = glbs[0];
            }

            res.BlendLesionPath = Path.Combine(meshesDir, "lesion_reconstructed.blend");
            if (!File.Exists(res.BlendLesionPath) && Directory.Exists(outputDir))
            {
                var blends = Directory.GetFiles(outputDir, "*.blend");
                if (blends.Length > 0) res.BlendLesionPath = blends[0];
            }

            res.MultiviewRenderPath = Path.Combine(figuresDir, "multiview_3d_renders.png");
            res.AnalysisPlotPath = Path.Combine(figuresDir, "reconstruction_analysis.png");
            res.SegmentationPlotPath = Path.Combine(figuresDir, "segmentation_stages.png");

            res.ReportJsonPath = Path.Combine(outputDir, "validation_report.json");
            res.ReportMdPath = Path.Combine(outputDir, "validation_report.md");

            if (File.Exists(res.ReportJsonPath))
            {
                try
                {
                    string json = File.ReadAllText(res.ReportJsonPath);
                    using var doc = JsonDocument.Parse(json);
                    var root = doc.RootElement;

                    if (root.TryGetProperty("segmentation", out var seg))
                    {
                        if (seg.TryGetProperty("cc", out var cc))
                        {
                            res.DiceCc = cc.GetProperty("dice_vs_manual").GetDouble();
                            res.IoUCc = cc.GetProperty("iou_vs_manual").GetDouble();
                        }
                        if (seg.TryGetProperty("mlo", out var mlo))
                        {
                            res.DiceMlo = mlo.GetProperty("dice_vs_manual").GetDouble();
                            res.IoUMlo = mlo.GetProperty("iou_vs_manual").GetDouble();
                        }
                    }

                    if (root.TryGetProperty("reconstruction", out var recon))
                    {
                        res.InitialMeanDevRatePercent = recon.GetProperty("initial_mean_deviation_rate_percent").GetDouble();
                        res.ArtMeanDevRatePercent = recon.GetProperty("art_mean_deviation_rate_percent").GetDouble();
                        res.FinalPvcMeanDevRatePercent = recon.GetProperty("final_pvc_mean_deviation_rate_percent").GetDouble();
                        if (recon.TryGetProperty("paper_benchmark_proposed_deviation_rate_percent", out var b))
                        {
                            res.PaperBenchmarkPercent = b.GetDouble();
                        }
                    }

                    if (root.TryGetProperty("mesh_properties", out var mesh))
                    {
                        res.VerticesCount = mesh.GetProperty("vertices").GetInt32();
                        res.FacesCount = mesh.GetProperty("faces").GetInt32();
                        res.VolumeMm3 = mesh.GetProperty("volume_mm3").GetDouble();
                        res.SurfaceAreaMm2 = mesh.GetProperty("surface_area_mm2").GetDouble();
                        res.Sphericity = mesh.GetProperty("sphericity").GetDouble();

                        if (mesh.TryGetProperty("triangulated_world_centroid_px", out var wc) && wc.GetArrayLength() == 3)
                        {
                            res.WorldCentroid = new[]
                            {
                                wc[0].GetDouble(),
                                wc[1].GetDouble(),
                                wc[2].GetDouble()
                            };
                        }
                    }
                }
                catch { }
            }

            return res;
        }
    }
}
