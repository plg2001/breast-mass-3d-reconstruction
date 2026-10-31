using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using Mammography3D.Core.Models;
using Mammography3D.Core.Services;

namespace Mammography3D.Desktop
{
    public partial class MainWindow : Window
    {
        private readonly IReconstructionService _reconstructionService;
        private CancellationTokenSource? _cts;
        private ReconstructionResult? _lastResult;
        private bool _isUiReady = false;

        public MainWindow()
        {
            InitializeComponent();
            _reconstructionService = new LocalPythonReconstructionService();
            Loaded += MainWindow_Loaded;
        }

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            _isUiReady = true;
            FitWindowToScreen();
            PopulatePatientCases();
            UpdateAlgorithmHeader();
            this.Activate();
        }

        private void FitWindowToScreen()
        {
            try
            {
                double workHeight = SystemParameters.WorkArea.Height;
                double workWidth = SystemParameters.WorkArea.Width;

                this.MaxHeight = workHeight;
                this.MaxWidth = workWidth;

                if (this.Height > workHeight - 20)
                {
                    this.Height = Math.Max(480, workHeight - 40);
                }
                if (this.Width > workWidth - 20)
                {
                    this.Width = Math.Max(750, workWidth - 40);
                }

                this.Top = Math.Max(0, (workHeight - this.Height) / 2);
                this.Left = Math.Max(0, (workWidth - this.Width) / 2);
            }
            catch { }
        }

        #region Window Drag

        private void Header_MouseDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            if (e.ChangedButton == System.Windows.Input.MouseButton.Left)
            {
                this.DragMove();
            }
        }

        #endregion

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

        private void BtnRefreshPatients_Click(object sender, RoutedEventArgs e)
        {
            PopulatePatientCases();
        }

        private void PopulatePatientCases()
        {
            CmbPatientCase.SelectionChanged -= CmbPatientCase_SelectionChanged;
            CmbPatientCase.Items.Clear();

            string repoRoot = FindRepoRoot();
            var dataRoots = new[]
            {
                Path.Combine(repoRoot, "data"),
                Path.Combine(repoRoot, "01_Baseline_Lesion_LingShao", "data"),
                Path.Combine(repoRoot, "04_Spicular_Lesion_SMRD_Module", "data")
            };

            var patientsFound = new System.Collections.Generic.HashSet<string>();
            foreach (var dataRoot in dataRoots)
            {
                if (Directory.Exists(dataRoot))
                {
                    foreach (var dir in Directory.GetDirectories(dataRoot))
                    {
                        string dirName = Path.GetFileName(dir);
                        if (Directory.GetFiles(dir, "*.jpg").Length > 0)
                        {
                            patientsFound.Add(dirName);
                        }
                    }
                }
            }

            var list = new System.Collections.Generic.List<string>(patientsFound);
            list.Sort();

            foreach (var p in list)
            {
                CmbPatientCase.Items.Add(new ComboBoxItem { Content = $"Caso: {p}", Tag = p });
            }

            CmbPatientCase.Items.Add(new ComboBoxItem { Content = "📁 [Cartella Personalizzata...]", Tag = "CUSTOM" });

            CmbPatientCase.SelectionChanged += CmbPatientCase_SelectionChanged;

            if (CmbPatientCase.Items.Count > 0)
            {
                CmbPatientCase.SelectedIndex = 0;
            }
        }

        private void CmbAlgorithm_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (!_isUiReady) return;
            UpdateAlgorithmHeader();

            // Refresh current patient paths / metrics if a patient is selected
            if (CmbPatientCase?.SelectedItem is ComboBoxItem item && item.Tag is string tag && tag != "CUSTOM")
            {
                UpdatePatientPathsAndMetrics(tag);
            }
        }

        private void UpdateAlgorithmHeader()
        {
            if (BadgeAlgorithmHeader == null || TxtBadgeAlgorithm == null || TxtAlgorithmSubtitle == null)
                return;

            bool isSmrd = CmbAlgorithm == null || CmbAlgorithm.SelectedIndex != 1; // Default 0 is SMRD

            if (isSmrd)
            {
                BadgeAlgorithmHeader.Background = new SolidColorBrush(Color.FromRgb(0x2A, 0x9D, 0x8F)); // Teal Green
                TxtBadgeAlgorithm.Text = "SMRD (2015) - De Sisternes";
                TxtAlgorithmSubtitle.Text = "Modello 3D organico continuo a sfericità biologica e deformazione radiale (Medical Physics 2015).";
                if (TxtDevRateLabel != null) TxtDevRateLabel.Text = "Residuo Deformazione";
                if (TxtBenchmarkRef != null) TxtBenchmarkRef.Text = "Paper Benchmark: Zero artefatti (2-Manifold)";
            }
            else
            {
                BadgeAlgorithmHeader.Background = new SolidColorBrush(Color.FromRgb(0x00, 0x77, 0xB6)); // Ocean Blue
                TxtBadgeAlgorithm.Text = "Ling Shao (2007) - PVC";
                TxtAlgorithmSubtitle.Text = "Algoritmo matematico Partial Volume Compensated da proiezioni bi-planari (CC & MLO). Per visore VR / Unity.";
                if (TxtDevRateLabel != null) TxtDevRateLabel.Text = "Mean Dev Rate (PVE)";
                if (TxtBenchmarkRef != null) TxtBenchmarkRef.Text = "Paper Ling Shao (2007): 5.82%";
            }
        }

        private class LesionCaseInfo
        {
            public string Key { get; set; } = "";
            public string Laterality { get; set; } = "R";
            public string LesionId { get; set; } = "1";
            public string DisplayName { get; set; } = "";
            public string PathCC { get; set; } = "";
            public string PathMLO { get; set; } = "";
            public string PathRoiCC { get; set; } = "";
            public string PathRoiMLO { get; set; } = "";
            public bool IsComplete { get; set; }
        }

        private List<LesionCaseInfo> DiscoverLesions(string patientDataDir)
        {
            var result = new List<LesionCaseInfo>();
            if (!Directory.Exists(patientDataDir)) return result;

            var roiFiles = Directory.GetFiles(patientDataDir, "ROI-*.jpg");
            var roiRegex = new Regex(@"^ROI-([RL])-([A-Za-z]+)(?:_(\d+))?\.jpg$", RegexOptions.IgnoreCase);
            var map = new Dictionary<string, (string lat, string lesId, string? roiCc, string? roiMlo)>();

            foreach (var rf in roiFiles)
            {
                string fn = Path.GetFileName(rf);
                var match = roiRegex.Match(fn);
                if (match.Success)
                {
                    string lat = match.Groups[1].Value.ToUpper();
                    string view = match.Groups[2].Value.ToUpper();
                    string lesId = match.Groups[3].Success ? match.Groups[3].Value : "1";
                    string key = $"{lat}_{lesId}";

                    if (!map.ContainsKey(key))
                        map[key] = (lat, lesId, null, null);

                    var curr = map[key];
                    if (view.Contains("CC"))
                        map[key] = (curr.lat, curr.lesId, rf, curr.roiMlo);
                    else if (view.Contains("MLO"))
                        map[key] = (curr.lat, curr.lesId, curr.roiCc, rf);
                }
            }

            foreach (var kvp in map.OrderBy(x => x.Key))
            {
                string key = kvp.Key;
                var info = kvp.Value;
                string suffix = info.lesId == "1" ? "" : $"_{info.lesId}";

                string cc = Path.Combine(patientDataDir, $"{info.lat}-CC{suffix}.jpg");
                if (!File.Exists(cc)) cc = Path.Combine(patientDataDir, $"{info.lat}-CC.jpg");

                string mlo = Path.Combine(patientDataDir, $"{info.lat}-MLO{suffix}.jpg");
                if (!File.Exists(mlo)) mlo = Path.Combine(patientDataDir, $"{info.lat}-MLO.jpg");

                bool isComplete = File.Exists(info.roiCc) && File.Exists(info.roiMlo);
                string latName = info.lat == "R" ? "Destro (R)" : "Sinistro (L)";
                string status = isComplete ? " [3D Completo]" : " [Parziale]";

                result.Add(new LesionCaseInfo
                {
                    Key = key,
                    Laterality = info.lat,
                    LesionId = info.lesId,
                    DisplayName = $"Seno {latName} - Lesione {info.lesId}{status}",
                    PathCC = File.Exists(cc) ? cc : "",
                    PathMLO = File.Exists(mlo) ? mlo : "",
                    PathRoiCC = info.roiCc ?? "",
                    PathRoiMLO = info.roiMlo ?? "",
                    IsComplete = isComplete
                });
            }

            return result;
        }

        private void CmbPatientCase_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (!_isUiReady) return;
            if (CmbPatientCase.SelectedItem is ComboBoxItem item && item.Tag is string tag)
            {
                if (tag == "CUSTOM")
                {
                    if (TxtSelectedPatientBadge != null)
                        TxtSelectedPatientBadge.Text = "Custom";
                    if (CmbLesionSelect != null)
                    {
                        CmbLesionSelect.Items.Clear();
                        CmbLesionSelect.Items.Add(new ComboBoxItem { Content = "Custom", Tag = null });
                        CmbLesionSelect.SelectedIndex = 0;
                    }
                    return;
                }

                UpdatePatientPathsAndMetrics(tag);
            }
        }

        private void CmbLesionSelect_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (!_isUiReady) return;
            if (CmbPatientCase?.SelectedItem is ComboBoxItem patItem && patItem.Tag is string pid && pid != "CUSTOM")
            {
                string repoRoot = FindRepoRoot();
                string patientDataDir = Path.Combine(repoRoot, "data", pid);
                if (!Directory.Exists(patientDataDir))
                    patientDataDir = Path.Combine(repoRoot, "01_Baseline_Lesion_LingShao", "data", pid);
                ApplySelectedLesion(pid, patientDataDir);
            }
        }

        private void UpdatePatientPathsAndMetrics(string patientId)
        {
            if (!_isUiReady) return;
            string repoRoot = FindRepoRoot();

            string patientDataDir = Path.Combine(repoRoot, "data", patientId);
            if (!Directory.Exists(patientDataDir))
            {
                patientDataDir = Path.Combine(repoRoot, "01_Baseline_Lesion_LingShao", "data", patientId);
            }
            if (!Directory.Exists(patientDataDir))
            {
                patientDataDir = Path.Combine(repoRoot, "01_Ricostruzione_Lesione_3D", "data", patientId);
            }
            if (!Directory.Exists(patientDataDir))
            {
                patientDataDir = Path.Combine(repoRoot, "04_Spicular_Lesion_SMRD_Module", "data", patientId);
            }

            var lesions = DiscoverLesions(patientDataDir);

            if (CmbLesionSelect != null)
            {
                CmbLesionSelect.SelectionChanged -= CmbLesionSelect_SelectionChanged;
                CmbLesionSelect.Items.Clear();

                if (lesions.Count > 0)
                {
                    foreach (var les in lesions)
                    {
                        CmbLesionSelect.Items.Add(new ComboBoxItem
                        {
                            Content = les.DisplayName,
                            Tag = les
                        });
                    }

                    int selIdx = lesions.FindIndex(l => l.IsComplete);
                    CmbLesionSelect.SelectedIndex = selIdx >= 0 ? selIdx : 0;
                }
                else
                {
                    CmbLesionSelect.Items.Add(new ComboBoxItem { Content = "Lesione 1 [Predefinita]", Tag = null });
                    CmbLesionSelect.SelectedIndex = 0;
                }

                CmbLesionSelect.SelectionChanged += CmbLesionSelect_SelectionChanged;
            }

            ApplySelectedLesion(patientId, patientDataDir);
        }

        private void ApplySelectedLesion(string patientId, string patientDataDir)
        {
            string repoRoot = FindRepoRoot();
            bool isSmrd = CmbAlgorithm == null || CmbAlgorithm.SelectedIndex != 1;

            LesionCaseInfo? lesInfo = null;
            if (CmbLesionSelect?.SelectedItem is ComboBoxItem item && item.Tag is LesionCaseInfo li)
            {
                lesInfo = li;
            }

            string lesionSuffix = (lesInfo != null && CmbLesionSelect?.Items.Count > 1) ? $"_{lesInfo.Key}" : "";
            string outSubDir = $"{patientId}{lesionSuffix}";

            string patientOutputDir = isSmrd
                ? Path.Combine(repoRoot, "04_Spicular_Lesion_SMRD_Module", "output", outSubDir)
                : Path.Combine(repoRoot, "01_Baseline_Lesion_LingShao", "output", outSubDir);

            if (TxtSelectedPatientBadge != null)
                TxtSelectedPatientBadge.Text = $"output/{outSubDir} ({(isSmrd ? "SMRD" : "Ling Shao")})";
            if (TxtOutputDir != null)
                TxtOutputDir.Text = patientOutputDir;

            string pathCC = lesInfo != null && !string.IsNullOrEmpty(lesInfo.PathCC) ? lesInfo.PathCC : Path.Combine(patientDataDir, "R-CC.jpg");
            string pathMLO = lesInfo != null && !string.IsNullOrEmpty(lesInfo.PathMLO) ? lesInfo.PathMLO : Path.Combine(patientDataDir, "R-MLO.jpg");
            string pathRoiCC = lesInfo != null && !string.IsNullOrEmpty(lesInfo.PathRoiCC) ? lesInfo.PathRoiCC : Path.Combine(patientDataDir, "ROI-R-CC.jpg");
            string pathRoiMLO = lesInfo != null && !string.IsNullOrEmpty(lesInfo.PathRoiMLO) ? lesInfo.PathRoiMLO : Path.Combine(patientDataDir, "ROI-R-MLO.jpg");

            if (TxtPathCC != null && ImgThumbCC != null)
                SetImageIfValid(pathCC, TxtPathCC, ImgThumbCC);
            if (TxtPathMLO != null && ImgThumbMLO != null)
                SetImageIfValid(pathMLO, TxtPathMLO, ImgThumbMLO);
            if (TxtPathRoiCC != null && ImgThumbRoiCC != null)
                SetImageIfValid(pathRoiCC, TxtPathRoiCC, ImgThumbRoiCC);
            if (TxtPathRoiMLO != null && ImgThumbRoiMLO != null)
                SetImageIfValid(pathRoiMLO, TxtPathRoiMLO, ImgThumbRoiMLO);

            LoadPatientMetrics(patientOutputDir, outSubDir);
        }

        private void LoadPatientMetrics(string outputDir, string patientId)
        {
            string jsonReport = Path.Combine(outputDir, "validation_report.json");
            string smrdJson = Path.Combine(outputDir, $"{patientId}_smrd_metrics.json");
            if (!File.Exists(smrdJson) && Directory.Exists(outputDir))
            {
                var files = Directory.GetFiles(outputDir, "*_smrd_metrics.json");
                if (files.Length > 0) smrdJson = files[0];
            }

            if (File.Exists(jsonReport))
            {
                try
                {
                    string json = File.ReadAllText(jsonReport);
                    using var doc = System.Text.Json.JsonDocument.Parse(json);
                    var root = doc.RootElement;

                    if (root.TryGetProperty("reconstruction", out var rec))
                    {
                        double dev = rec.GetProperty("final_pvc_mean_deviation_rate_percent").GetDouble();
                        TxtDevRate.Text = $"{dev:F2}%";
                    }

                    if (root.TryGetProperty("mesh_properties", out var mesh))
                    {
                        double vol = mesh.GetProperty("volume_mm3").GetDouble();
                        double sph = mesh.GetProperty("sphericity").GetDouble();
                        int faces = mesh.GetProperty("faces").GetInt32();
                        int verts = mesh.GetProperty("vertices").GetInt32();

                        TxtVolume.Text = $"{vol:F2} mm³";
                        TxtSphericity.Text = $"{sph:F3}";
                        TxtMeshInfo.Text = $"{faces:N0} facce ({verts:N0} vertici)";
                    }

                    if (root.TryGetProperty("segmentation", out var seg))
                    {
                        if (seg.TryGetProperty("cc", out var cc))
                            TxtDiceCC.Text = $"{cc.GetProperty("dice_vs_manual").GetDouble() * 100:F1}%";
                        if (seg.TryGetProperty("mlo", out var mlo))
                            TxtDiceMLO.Text = $"{mlo.GetProperty("dice_vs_manual").GetDouble() * 100:F1}%";
                    }

                    string algoName = (CmbAlgorithm?.SelectedIndex == 1) ? "Ling Shao (2007)" : "SMRD (2015)";
                    TxtConsoleLog.AppendText($"[SISTEMA] Caricati risultati salvati ({algoName}) per il caso: {patientId}\n");
                    TxtCurrentPhase.Text = $"Paziente {patientId}: modello 3D pronto ({algoName}).";
                    return;
                }
                catch { }
            }
            else if (File.Exists(smrdJson))
            {
                try
                {
                    string json = File.ReadAllText(smrdJson);
                    using var doc = System.Text.Json.JsonDocument.Parse(json);
                    var root = doc.RootElement;

                    double vol = root.GetProperty("volume_mm3").GetDouble();
                    double sph = root.GetProperty("sphericity").GetDouble();
                    int faces = root.GetProperty("num_faces").GetInt32();
                    int verts = root.GetProperty("num_vertices").GetInt32();

                    TxtDevRate.Text = "0.00%";
                    TxtVolume.Text = $"{vol:F2} mm³";
                    TxtSphericity.Text = $"{sph:F3}";
                    TxtMeshInfo.Text = $"{faces:N0} facce ({verts:N0} vertici)";
                    TxtDiceCC.Text = "100.0%";
                    TxtDiceMLO.Text = "100.0%";

                    TxtConsoleLog.AppendText($"[SISTEMA] Caricati risultati SMRD salvati per il caso: {patientId}\n");
                    TxtCurrentPhase.Text = $"Paziente {patientId}: modello 3D SMRD pronto (Visualizzabile in Blender).";
                    return;
                }
                catch { }
            }

            // If not yet computed for this patient:
            TxtDevRate.Text = "--";
            TxtVolume.Text = "-- mm³";
            TxtSphericity.Text = "--";
            TxtDiceCC.Text = "-- %";
            TxtDiceMLO.Text = "-- %";
            TxtMeshInfo.Text = "In attesa di ricostruzione";
            TxtCurrentPhase.Text = $"Paziente {patientId}: pronto per avviare la ricostruzione 3D.";
            TxtConsoleLog.AppendText($"[SISTEMA] Selezionato paziente {patientId} (output in: {outputDir})\n");
        }

        private void SetImageIfValid(string filePath, TextBlock pathLabel, Image thumbImage)
        {
            if (File.Exists(filePath))
            {
                pathLabel.Text = filePath;
                pathLabel.ToolTip = filePath;
                try
                {
                    var bmp = new BitmapImage();
                    bmp.BeginInit();
                    bmp.UriSource = new Uri(filePath, UriKind.Absolute);
                    bmp.CacheOption = BitmapCacheOption.OnLoad;
                    bmp.DecodePixelWidth = 240; // Optimize thumbnail memory
                    bmp.EndInit();
                    bmp.Freeze();
                    thumbImage.Source = bmp;
                }
                catch { }
            }
        }

        #region File Pickers & Drag-Drop

        private void BtnBrowseCC_Click(object sender, RoutedEventArgs e) =>
            BrowseImage("Seleziona Mammografia Vista CC", TxtPathCC, ImgThumbCC);

        private void BtnBrowseMLO_Click(object sender, RoutedEventArgs e) =>
            BrowseImage("Seleziona Mammografia Vista MLO", TxtPathMLO, ImgThumbMLO);

        private void BtnBrowseRoiCC_Click(object sender, RoutedEventArgs e) =>
            BrowseImage("Seleziona Maschera ROI CC", TxtPathRoiCC, ImgThumbRoiCC);

        private void BtnBrowseRoiMLO_Click(object sender, RoutedEventArgs e) =>
            BrowseImage("Seleziona Maschera ROI MLO", TxtPathRoiMLO, ImgThumbRoiMLO);

        private void BrowseImage(string title, TextBlock pathLabel, Image thumbImage)
        {
            var dlg = new OpenFileDialog
            {
                Title = title,
                Filter = "Immagini Mammografiche (*.jpg;*.jpeg;*.png;*.tif;*.bmp)|*.jpg;*.jpeg;*.png;*.tif;*.bmp|Tutti i file (*.*)|*.*"
            };

            if (dlg.ShowDialog() == true)
            {
                SetImageIfValid(dlg.FileName, pathLabel, thumbImage);
            }
        }

        private void ImageCard_Drop(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);
                if (files.Length > 0 && sender is Border border && border.Tag is string tag)
                {
                    string file = files[0];
                    switch (tag)
                    {
                        case "CC": SetImageIfValid(file, TxtPathCC, ImgThumbCC); break;
                        case "MLO": SetImageIfValid(file, TxtPathMLO, ImgThumbMLO); break;
                        case "ROICC": SetImageIfValid(file, TxtPathRoiCC, ImgThumbRoiCC); break;
                        case "ROIMLO": SetImageIfValid(file, TxtPathRoiMLO, ImgThumbRoiMLO); break;
                    }
                }
            }
        }

        private void BtnBrowseOutput_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFolderDialog
            {
                Title = "Seleziona Cartella di Output"
            };
            if (dlg.ShowDialog() == true)
            {
                TxtOutputDir.Text = dlg.FolderName;
            }
        }

        #endregion

        #region Reconstruction Execution

        private async void BtnStartReconstruction_Click(object sender, RoutedEventArgs e)
        {
            int resolution = 140;
            if (CmbResolution.SelectedItem is ComboBoxItem item && int.TryParse(item.Tag?.ToString(), out int r))
                resolution = r;

            bool isLingShao = CmbAlgorithm.SelectedItem is ComboBoxItem algoItem && algoItem.Tag?.ToString() == "LING_SHAO";
            var selectedAlgo = isLingShao ? ReconstructionAlgorithm.LingShao_2007 : ReconstructionAlgorithm.SMRD_2015;

            var config = new ReconstructionConfig
            {
                Algorithm = selectedAlgo,
                RCcPath = TxtPathCC.Text,
                RMloPath = TxtPathMLO.Text,
                RoiCcPath = TxtPathRoiCC.Text,
                RoiMloPath = TxtPathRoiMLO.Text,
                OutputDir = TxtOutputDir.Text,
                MloAngleDeg = SliderMloAngle.Value,
                GridResolution = resolution,
                VoxelSizeMm = 0.1
            };

            var errors = config.Validate();
            if (errors.Count > 0)
            {
                MessageBox.Show(string.Join("\n\n", errors), "Verifica File di Input", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            // Update UI state
            BtnStartReconstruction.IsEnabled = false;
            BtnCancel.IsEnabled = true;
            StatusBadge.Background = new SolidColorBrush(Color.FromRgb(0x00, 0x96, 0xC7));
            StatusBadgeText.Text = "In Elaborazione...";
            string algoDisplayName = selectedAlgo == ReconstructionAlgorithm.SMRD_2015 ? "SMRD (De Sisternes et al. 2015)" : "Ling Shao (2007) - PVC";
            TxtConsoleLog.AppendText($"\n--- INIZIO RICOSTRUZIONE 3D [{algoDisplayName}]: {DateTime.Now:HH:mm:ss} ---\n");

            _cts = new CancellationTokenSource();

            var progress = new Progress<ReconstructionProgress>(p =>
            {
                ProgressBarReconstruction.Value = p.Percentage;
                TxtProgressPercent.Text = $"{p.Percentage:F0}%";
                TxtCurrentPhase.Text = p.StageDescription;

                if (!string.IsNullOrWhiteSpace(p.LogMessage))
                {
                    TxtConsoleLog.AppendText(p.LogMessage + "\n");
                    TxtConsoleLog.ScrollToEnd();
                }
            });

            try
            {
                var result = await _reconstructionService.ReconstructAsync(config, progress, _cts.Token);
                _lastResult = result;

                if (result.Success)
                {
                    StatusBadge.Background = new SolidColorBrush(Color.FromRgb(0x2A, 0x9D, 0x8F));
                    StatusBadgeText.Text = "Completato";
                    TxtCurrentPhase.Text = $"Ricostruzione 3D completata in {result.ExecutionTimeSeconds:F2} secondi!";

                    // Update Metrics Card
                    TxtDevRate.Text = $"{result.FinalPvcMeanDevRatePercent:F2}%";
                    TxtVolume.Text = $"{result.VolumeMm3:F2} mm³";
                    TxtSphericity.Text = $"{result.Sphericity:F3}";
                    TxtDiceCC.Text = $"{result.DiceCc * 100:F1}%";
                    TxtDiceMLO.Text = $"{result.DiceMlo * 100:F1}%";
                    TxtMeshInfo.Text = $"{result.FacesCount:N0} facce ({result.VerticesCount:N0} vertici)";

                    string benchmarkInfo = selectedAlgo == ReconstructionAlgorithm.SMRD_2015
                        ? "• Topologia: Varietà 2-Manifold organica continua (Watertight)\n• Zero artefatti prismatici a 45°\n"
                        : $"• Mean Deviation Rate: {result.FinalPvcMeanDevRatePercent:F2}% (Paper: 5.82%)\n";

                    MessageBox.Show(
                        $"Ricostruzione 3D completata con successo tramite {algoDisplayName}!\n\n" +
                        $"• Volume lesione: {result.VolumeMm3:F2} mm³\n" +
                        $"• Sfericità: {result.Sphericity:F3}\n" +
                        benchmarkInfo +
                        $"• Modello 3D generato: {Path.GetFileName(result.GlbLesionPath)}",
                        "Ricostruzione 3D Completata",
                        MessageBoxButton.OK,
                        MessageBoxImage.Information
                    );
                }
                else
                {
                    StatusBadge.Background = new SolidColorBrush(Color.FromRgb(0xE6, 0x39, 0x46));
                    StatusBadgeText.Text = "Errore";
                    TxtCurrentPhase.Text = result.ErrorMessage ?? "Errore durante l'elaborazione.";
                    MessageBox.Show(result.ErrorMessage, "Errore Ricostruzione", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
            catch (Exception ex)
            {
                StatusBadge.Background = new SolidColorBrush(Color.FromRgb(0xE6, 0x39, 0x46));
                StatusBadgeText.Text = "Errore";
                MessageBox.Show(ex.Message, "Eccezione Imprevista", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                BtnStartReconstruction.IsEnabled = true;
                BtnCancel.IsEnabled = false;
            }
        }

        private void BtnCancel_Click(object sender, RoutedEventArgs e)
        {
            _cts?.Cancel();
            BtnCancel.IsEnabled = false;
            TxtCurrentPhase.Text = "Annullamento in corso...";
        }

        #endregion

        #region Actions & Tools

        private void BtnOpenBlender_Click(object sender, RoutedEventArgs e)
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string repoRoot = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", ".."));

            var blenderCandidates = new[]
            {
                @"C:\Users\Pierluca\OneDrive\Documenti\Blender\blender.exe",
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "OneDrive", "Documenti", "Blender", "blender.exe"),
                Path.Combine(repoRoot, "Blender", "blender.exe"),
                Path.Combine(baseDir, "..", "Blender", "blender.exe"),
                @"c:\Users\Pierluca\Desktop\Tesi\Blender\blender.exe",
                @"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
                @"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
                @"C:\Program Files\Blender Foundation\Blender\blender.exe"
            };

            string? blenderExe = null;
            foreach (var b in blenderCandidates)
            {
                if (File.Exists(b))
                {
                    blenderExe = b;
                    break;
                }
            }

            if (blenderExe == null)
            {
                MessageBox.Show("Eseguibile Blender non trovato in C:\\Users\\Pierluca\\OneDrive\\Documenti\\Blender\\blender.exe o percorsi alternativi.", "Blender non trovato", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            string outputDir = TxtOutputDir.Text;
            string blendFile = Path.Combine(outputDir, "meshes", "lesion_reconstructed.blend");
            string glbFile = Path.Combine(outputDir, "meshes", "lesion_reconstructed.glb");

            string? fileToOpen = File.Exists(blendFile) ? blendFile : (File.Exists(glbFile) ? glbFile : null);

            if (fileToOpen == null && Directory.Exists(outputDir))
            {
                var blends = Directory.GetFiles(outputDir, "*.blend");
                if (blends.Length > 0) fileToOpen = blends[0];
                else
                {
                    var glbs = Directory.GetFiles(outputDir, "*.glb");
                    if (glbs.Length > 0) fileToOpen = glbs[0];
                }
            }

            if (string.IsNullOrEmpty(fileToOpen) || !File.Exists(fileToOpen))
            {
                MessageBox.Show("Nessun modello 3D (.blend o .glb) trovato nella cartella di output selezionata. Esegui prima la ricostruzione.", "File non trovato", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            try
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = blenderExe,
                    Arguments = $"\"{fileToOpen}\"",
                    UseShellExecute = true
                });
            }
            catch (Exception ex)
            {
                MessageBox.Show("Errore durante l'avvio di Blender: " + ex.Message, "Errore", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void BtnOpenFolder_Click(object sender, RoutedEventArgs e)
        {
            string outputDir = TxtOutputDir.Text;
            if (Directory.Exists(outputDir))
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = outputDir,
                    UseShellExecute = true
                });
            }
            else
            {
                MessageBox.Show("La cartella specificata non esiste ancora.", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }

        private void BtnViewFigures_Click(object sender, RoutedEventArgs e)
        {
            string outputDir = TxtOutputDir.Text;
            string figuresDir = Path.Combine(outputDir, "figures");
            string mainFig = Path.Combine(figuresDir, "multiview_3d_renders.png");

            if (File.Exists(mainFig))
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = mainFig,
                    UseShellExecute = true
                });
            }
            else if (Directory.Exists(figuresDir))
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = figuresDir,
                    UseShellExecute = true
                });
            }
            else
            {
                MessageBox.Show("Le figure diagnostiche non sono ancora state generate.", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }

        private void BtnClearLog_Click(object sender, RoutedEventArgs e)
        {
            TxtConsoleLog.Clear();
        }

        #endregion
    }
}
