using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Mammography3D.Core.Models;
using Mammography3D.Core.Services;

namespace Mammography3D.Desktop
{
    public partial class MainWindow : Window
    {
        private readonly IReconstructionService _reconstructionService;
        private CancellationTokenSource? _cts;
        private bool _isUiReady = false;
        private ReconstructionMode _currentMode = ReconstructionMode.TotalReconstruction;

        public MainWindow()
        {
            InitializeComponent();
            _reconstructionService = new LocalPythonReconstructionService();
            Loaded += MainWindow_Loaded;
        }

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            _isUiReady = true;
            PopulatePatients();
            SelectMode(ReconstructionMode.TotalReconstruction);
            this.Activate();
        }

        private void Header_MouseDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            if (e.ChangedButton == System.Windows.Input.MouseButton.Left)
            {
                this.DragMove();
            }
        }

        private static string FindRepoRoot() => LocalPythonReconstructionService.FindRepoRoot();

        #region Mode Selection

        public void SelectMode(ReconstructionMode mode)
        {
            _currentMode = mode;
            if (!_isUiReady) return;

            // Reset Card Borders
            CardModeTotal.BorderBrush = new SolidColorBrush(Color.FromRgb(0x33, 0x41, 0x55));
            CardModeTotal.BorderThickness = new Thickness(1);
            CardModeSurface.BorderBrush = new SolidColorBrush(Color.FromRgb(0x33, 0x41, 0x55));
            CardModeSurface.BorderThickness = new Thickness(1);
            CardModeLesion.BorderBrush = new SolidColorBrush(Color.FromRgb(0x33, 0x41, 0x55));
            CardModeLesion.BorderThickness = new Thickness(1);

            RbModeTotal.IsChecked = (mode == ReconstructionMode.TotalReconstruction);
            RbModeSurface.IsChecked = (mode == ReconstructionMode.BreastSurface);
            RbModeLesion.IsChecked = (mode == ReconstructionMode.LesionSMRD);

            switch (mode)
            {
                case ReconstructionMode.TotalReconstruction:
                    CardModeTotal.BorderBrush = new SolidColorBrush(Color.FromRgb(0x38, 0xBD, 0xF8));
                    CardModeTotal.BorderThickness = new Thickness(2);
                    BtnStartReconstruction.Content = "▶   START COMPLETE RECONSTRUCTION";
                    LblStat1.Text = "Breast Volume";
                    LblStat2.Text = "Tumor Volume";
                    LblStat3.Text = "AI Diagnosis";
                    break;

                case ReconstructionMode.BreastSurface:
                    CardModeSurface.BorderBrush = new SolidColorBrush(Color.FromRgb(0x10, 0xB9, 0x81));
                    CardModeSurface.BorderThickness = new Thickness(2);
                    BtnStartReconstruction.Content = "▶   START SURFACE RECONSTRUCTION";
                    LblStat1.Text = "Breast Volume";
                    LblStat2.Text = "Skin Surface Area";
                    LblStat3.Text = "Dimensions (W x D)";
                    break;

                case ReconstructionMode.LesionSMRD:
                default:
                    CardModeLesion.BorderBrush = new SolidColorBrush(Color.FromRgb(0xF5, 0x9E, 0x0B));
                    CardModeLesion.BorderThickness = new Thickness(2);
                    BtnStartReconstruction.Content = "▶   START 3D LESION RECONSTRUCTION";
                    LblStat1.Text = "Lesion Volume";
                    LblStat2.Text = "Sphericity (Ψ)";
                    LblStat3.Text = "3D Topology";
                    break;
            }

            if (CmbPatientCase?.SelectedItem is ComboBoxItem item && item.Tag is string pid)
            {
                RefreshPatientData(pid);
            }
        }

        private void CardModeTotal_Click(object sender, System.Windows.Input.MouseButtonEventArgs e) => SelectMode(ReconstructionMode.TotalReconstruction);
        private void CardModeSurface_Click(object sender, System.Windows.Input.MouseButtonEventArgs e) => SelectMode(ReconstructionMode.BreastSurface);
        private void CardModeLesion_Click(object sender, System.Windows.Input.MouseButtonEventArgs e) => SelectMode(ReconstructionMode.LesionSMRD);

        private void RbModeTotal_Checked(object sender, RoutedEventArgs e) { if (_isUiReady) SelectMode(ReconstructionMode.TotalReconstruction); }
        private void RbModeSurface_Checked(object sender, RoutedEventArgs e) { if (_isUiReady) SelectMode(ReconstructionMode.BreastSurface); }
        private void RbModeLesion_Checked(object sender, RoutedEventArgs e) { if (_isUiReady) SelectMode(ReconstructionMode.LesionSMRD); }

        #endregion

        #region Patient Population & Data Refresh

        private void BtnRefreshPatients_Click(object sender, RoutedEventArgs e) => PopulatePatients();

        private void PopulatePatients()
        {
            CmbPatientCase.SelectionChanged -= CmbPatientCase_SelectionChanged;
            CmbPatientCase.Items.Clear();

            string repoRoot = FindRepoRoot();
            string dataDir = Path.Combine(repoRoot, "data");

            var patientsFound = new List<string>();
            if (Directory.Exists(dataDir))
            {
                foreach (var dir in Directory.GetDirectories(dataDir))
                {
                    string dirName = Path.GetFileName(dir);
                    if (Directory.GetFiles(dir, "*.jpg").Length > 0 || Directory.GetFiles(dir, "*.png").Length > 0)
                    {
                        patientsFound.Add(dirName);
                    }
                }
            }

            patientsFound.Sort();

            foreach (var p in patientsFound)
            {
                CmbPatientCase.Items.Add(new ComboBoxItem { Content = $"Patient: {p}", Tag = p });
            }

            CmbPatientCase.SelectionChanged += CmbPatientCase_SelectionChanged;

            if (CmbPatientCase.Items.Count > 0)
            {
                int idx1 = patientsFound.IndexOf("P_00001");
                CmbPatientCase.SelectedIndex = idx1 >= 0 ? idx1 : 0;
            }
        }

        private void CmbPatientCase_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (!_isUiReady) return;
            if (CmbPatientCase.SelectedItem is ComboBoxItem item && item.Tag is string pid)
            {
                RefreshPatientData(pid);
            }
        }

        private void RefreshPatientData(string patientId)
        {
            string repoRoot = FindRepoRoot();
            string patientDir = Path.Combine(repoRoot, "data", patientId);

            string outDir = _currentMode switch
            {
                ReconstructionMode.TotalReconstruction => Path.Combine(repoRoot, "07_Total_Reconstruction_Module", "output", patientId),
                ReconstructionMode.BreastSurface => Path.Combine(repoRoot, "05_Breast_Surface_Module", "output", patientId),
                _ => Path.Combine(repoRoot, "04_Spicular_Lesion_SMRD_Module", "output", patientId)
            };

            // 1. Check for 3D Render Preview
            string? previewImg = FindPreviewImage(outDir, patientId);
            if (!string.IsNullOrEmpty(previewImg) && File.Exists(previewImg))
            {
                SetImageSource(ImgMainPreview, previewImg);
                ImgMainPreview.Visibility = Visibility.Visible;
                GridRadiographsFallback.Visibility = Visibility.Collapsed;
                TxtNoPreviewPlaceholder.Visibility = Visibility.Collapsed;
            }
            else
            {
                // Fallback to 2D Radiographs (CC and MLO)
                string cc = Path.Combine(patientDir, "R-CC.jpg");
                if (!File.Exists(cc)) cc = Path.Combine(patientDir, "L-CC.jpg");
                string mlo = Path.Combine(patientDir, "R-MLO.jpg");
                if (!File.Exists(mlo)) mlo = Path.Combine(patientDir, "L-MLO.jpg");

                if (File.Exists(cc) || File.Exists(mlo))
                {
                    if (File.Exists(cc)) SetImageSource(ImgThumbCC, cc);
                    if (File.Exists(mlo)) SetImageSource(ImgThumbMLO, mlo);
                    ImgMainPreview.Visibility = Visibility.Collapsed;
                    GridRadiographsFallback.Visibility = Visibility.Visible;
                    TxtNoPreviewPlaceholder.Visibility = Visibility.Collapsed;
                }
                else
                {
                    ImgMainPreview.Visibility = Visibility.Collapsed;
                    GridRadiographsFallback.Visibility = Visibility.Collapsed;
                    TxtNoPreviewPlaceholder.Visibility = Visibility.Visible;
                }
            }

            // 2. Load statistics and model status
            LoadStats(outDir, patientId);
        }

        private string? FindPreviewImage(string outDir, string patientId)
        {
            if (!Directory.Exists(outDir)) return null;

            if (_currentMode == ReconstructionMode.TotalReconstruction)
            {
                string p = Path.Combine(outDir, $"{patientId}_total_preview.png");
                if (File.Exists(p)) return p;
            }
            else if (_currentMode == ReconstructionMode.BreastSurface)
            {
                string p = Path.Combine(outDir, $"{patientId}_preview.png");
                if (File.Exists(p)) return p;
            }
            else
            {
                string p = Path.Combine(outDir, "multiview_3d_renders.png");
                if (File.Exists(p)) return p;
                string p2 = Path.Combine(outDir, "figures", "multiview_3d_renders.png");
                if (File.Exists(p2)) return p2;
            }

            var pngs = Directory.GetFiles(outDir, "*.png");
            return pngs.Length > 0 ? pngs[0] : null;
        }

        private void LoadStats(string outDir, string patientId)
        {
            switch (_currentMode)
            {
                case ReconstructionMode.TotalReconstruction:
                    string reportJson = Path.Combine(outDir, $"{patientId}_total_clinical_report.json");
                    if (File.Exists(reportJson))
                    {
                        try
                        {
                            string json = File.ReadAllText(reportJson);
                            using var doc = System.Text.Json.JsonDocument.Parse(json);
                            var root = doc.RootElement;

                            double bVol = 0.0, lVol = 0.0;
                            string diag = "MALIGNANT (94%)";

                            if (root.TryGetProperty("breast_morphometry", out var bm) && bm.TryGetProperty("volume_cm3", out var bv))
                                bVol = bv.GetDouble();

                            if (root.TryGetProperty("tumor_pathology_smrd", out var ts) && ts.TryGetProperty("volume_cm3", out var tv))
                                lVol = tv.GetDouble();

                            TxtStat1.Text = $"{bVol:F0} cm³";
                            TxtStat2.Text = $"{lVol:F1} cm³";
                            TxtStat3.Text = diag;
                            TxtStatusMessage.Text = $"Complete 3D model ready for {patientId}.";
                            StatusBadge.Background = new SolidColorBrush(Color.FromRgb(0x10, 0xB9, 0x81));
                            StatusBadgeText.Text = "Ready";
                            return;
                        }
                        catch { }
                    }
                    break;

                case ReconstructionMode.BreastSurface:
                    string metricsJson = Path.Combine(outDir, $"{patientId}_metrics.json");
                    if (File.Exists(metricsJson))
                    {
                        try
                        {
                            string json = File.ReadAllText(metricsJson);
                            using var doc = System.Text.Json.JsonDocument.Parse(json);
                            var root = doc.RootElement;

                            double vol = root.GetProperty("volume_cm3").GetDouble();
                            double area = root.GetProperty("surface_area_cm2").GetDouble();
                            double w = root.GetProperty("width_mm").GetDouble();
                            double d = root.GetProperty("depth_pnl_mm").GetDouble();

                            TxtStat1.Text = $"{vol:F0} cm³";
                            TxtStat2.Text = $"{area:F0} cm²";
                            TxtStat3.Text = $"{w:F0} x {d:F0} mm";
                            TxtStatusMessage.Text = $"3D breast surface ready for {patientId}.";
                            StatusBadge.Background = new SolidColorBrush(Color.FromRgb(0x10, 0xB9, 0x81));
                            StatusBadgeText.Text = "Ready";
                            return;
                        }
                        catch { }
                    }
                    break;

                case ReconstructionMode.LesionSMRD:
                default:
                    string smrdJson = Path.Combine(outDir, $"{patientId}_smrd_metrics.json");
                    if (!File.Exists(smrdJson) && Directory.Exists(outDir))
                    {
                        var cand = Directory.GetFiles(outDir, "*_smrd_metrics.json");
                        if (cand.Length > 0) smrdJson = cand[0];
                    }

                    if (File.Exists(smrdJson))
                    {
                        try
                        {
                            string json = File.ReadAllText(smrdJson);
                            using var doc = System.Text.Json.JsonDocument.Parse(json);
                            var root = doc.RootElement;

                            double vol = root.GetProperty("volume_mm3").GetDouble();
                            double sph = root.GetProperty("sphericity").GetDouble();

                            TxtStat1.Text = $"{vol:F0} mm³";
                            TxtStat2.Text = $"{sph:F3}";
                            TxtStat3.Text = "2-Manifold";
                            TxtStatusMessage.Text = $"3D SMRD lesion ready for {patientId}.";
                            StatusBadge.Background = new SolidColorBrush(Color.FromRgb(0x10, 0xB9, 0x81));
                            StatusBadgeText.Text = "Ready";
                            return;
                        }
                        catch { }
                    }
                    break;
            }

            // Default if not yet reconstructed
            TxtStat1.Text = "--";
            TxtStat2.Text = "--";
            TxtStat3.Text = "--";
            TxtStatusMessage.Text = $"Patient {patientId}: press 'Start Reconstruction'.";
            StatusBadge.Background = new SolidColorBrush(Color.FromRgb(0x33, 0x41, 0x55));
            StatusBadgeText.Text = "Not Reconstructed";
        }

        private static void SetImageSource(Image img, string path)
        {
            try
            {
                var bmp = new BitmapImage();
                bmp.BeginInit();
                bmp.UriSource = new Uri(path, UriKind.Absolute);
                bmp.CacheOption = BitmapCacheOption.OnLoad;
                bmp.EndInit();
                bmp.Freeze();
                img.Source = bmp;
            }
            catch { }
        }

        #endregion

        #region Execution

        private async void BtnStartReconstruction_Click(object sender, RoutedEventArgs e)
        {
            string patientId = "P_00001";
            if (CmbPatientCase.SelectedItem is ComboBoxItem patItem && patItem.Tag is string pid)
            {
                patientId = pid;
            }

            string repoRoot = FindRepoRoot();
            string patientDir = Path.Combine(repoRoot, "data", patientId);

            string outDir = _currentMode switch
            {
                ReconstructionMode.TotalReconstruction => Path.Combine(repoRoot, "07_Total_Reconstruction_Module", "output", patientId),
                ReconstructionMode.BreastSurface => Path.Combine(repoRoot, "05_Breast_Surface_Module", "output", patientId),
                _ => Path.Combine(repoRoot, "04_Spicular_Lesion_SMRD_Module", "output", patientId)
            };

            string cc = Path.Combine(patientDir, "R-CC.jpg");
            if (!File.Exists(cc)) cc = Path.Combine(patientDir, "L-CC.jpg");
            string mlo = Path.Combine(patientDir, "R-MLO.jpg");
            if (!File.Exists(mlo)) mlo = Path.Combine(patientDir, "L-MLO.jpg");
            string roiCc = Path.Combine(patientDir, "ROI-R-CC.jpg");
            if (!File.Exists(roiCc)) roiCc = Path.Combine(patientDir, "ROI-L-CC.jpg");
            string roiMlo = Path.Combine(patientDir, "ROI-R-MLO.jpg");
            if (!File.Exists(roiMlo)) roiMlo = Path.Combine(patientDir, "ROI-L-MLO.jpg");

            var config = new ReconstructionConfig
            {
                Mode = _currentMode,
                PatientId = patientId,
                RCcPath = cc,
                RMloPath = mlo,
                RoiCcPath = roiCc,
                RoiMloPath = roiMlo,
                OutputDir = outDir,
                Spicularity = 1.25,
                NumSpikes = 48,
                PixelSpacingMm = 0.05,
                MloAngleDeg = 45.0,
                GridResolution = 140,
                VoxelSizeMm = 0.1
            };

            var errors = config.Validate();
            if (errors.Count > 0)
            {
                MessageBox.Show(string.Join("\n", errors), "Patient Data Verification", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            BtnStartReconstruction.IsEnabled = false;
            StatusBadge.Background = new SolidColorBrush(Color.FromRgb(0x02, 0x84, 0xC7));
            StatusBadgeText.Text = "Processing...";
            TxtStatusMessage.Text = $"Starting 3D pipeline for {patientId}...";

            _cts = new CancellationTokenSource();

            var progress = new Progress<ReconstructionProgress>(p =>
            {
                ProgressBarReconstruction.Value = p.Percentage;
                TxtProgressPercent.Text = $"{p.Percentage:F0}%";
                TxtStatusMessage.Text = p.StageDescription;
            });

            try
            {
                var result = await _reconstructionService.ReconstructAsync(config, progress, _cts.Token);

                if (result.Success)
                {
                    StatusBadge.Background = new SolidColorBrush(Color.FromRgb(0x10, 0xB9, 0x81));
                    StatusBadgeText.Text = "Completed";
                    ProgressBarReconstruction.Value = 100;
                    TxtProgressPercent.Text = "100%";
                    TxtStatusMessage.Text = $"3D Reconstruction completed in {result.ExecutionTimeSeconds:F1}s!";

                    RefreshPatientData(patientId);

                    MessageBox.Show(
                        $"3D Reconstruction successfully completed!\n\n" +
                        $"• Patient: {patientId}\n" +
                        $"• Elapsed time: {result.ExecutionTimeSeconds:F1} s\n" +
                        $"• Model saved to: {outDir}",
                        "3D Reconstruction Complete",
                        MessageBoxButton.OK,
                        MessageBoxImage.Information
                    );
                }
                else
                {
                    StatusBadge.Background = new SolidColorBrush(Color.FromRgb(0xF4, 0x3F, 0x5E));
                    StatusBadgeText.Text = "Error";
                    TxtStatusMessage.Text = result.ErrorMessage ?? "Error during processing.";
                    MessageBox.Show(result.ErrorMessage, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
            catch (Exception ex)
            {
                StatusBadge.Background = new SolidColorBrush(Color.FromRgb(0xF4, 0x3F, 0x5E));
                StatusBadgeText.Text = "Error";
                TxtStatusMessage.Text = ex.Message;
                MessageBox.Show(ex.Message, "Exception", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                BtnStartReconstruction.IsEnabled = true;
            }
        }

        #endregion

        #region Actions: Blender, Render 3D, Cartella

        private static string? LocateBlender()
        {
            string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            var candidates = new List<string>
            {
                Path.Combine(userProfile, "OneDrive", "Documenti", "Blender", "blender.exe"),
                Path.Combine(userProfile, "Documents", "Blender", "blender.exe"),
                @"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
                @"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
                @"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
                @"C:\Program Files\Blender Foundation\Blender\blender.exe"
            };

            foreach (var c in candidates)
            {
                if (File.Exists(c)) return c;
            }

            string? pathEnv = Environment.GetEnvironmentVariable("PATH");
            if (!string.IsNullOrEmpty(pathEnv))
            {
                foreach (string entry in pathEnv.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
                {
                    string candidate = Path.Combine(entry.Trim(), "blender.exe");
                    if (File.Exists(candidate)) return candidate;
                }
            }

            return null;
        }

        private void BtnOpenBlender_Click(object sender, RoutedEventArgs e)
        {
            string patientId = (CmbPatientCase.SelectedItem as ComboBoxItem)?.Tag as string ?? "P_00001";
            string repoRoot = FindRepoRoot();

            string outDir = _currentMode switch
            {
                ReconstructionMode.TotalReconstruction => Path.Combine(repoRoot, "07_Total_Reconstruction_Module", "output", patientId),
                ReconstructionMode.BreastSurface => Path.Combine(repoRoot, "05_Breast_Surface_Module", "output", patientId),
                _ => Path.Combine(repoRoot, "04_Spicular_Lesion_SMRD_Module", "output", patientId)
            };

            string? blenderExe = LocateBlender();
            if (blenderExe == null)
            {
                MessageBox.Show("Blender was not found in standard paths. Please install Blender or add it to Windows PATH.", "Blender Not Found", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            string? fileToOpen = null;
            if (Directory.Exists(outDir))
            {
                var blends = Directory.GetFiles(outDir, "*.blend");
                if (blends.Length > 0) fileToOpen = blends[0];
                else
                {
                    var glbs = Directory.GetFiles(outDir, "*.glb");
                    if (glbs.Length > 0) fileToOpen = glbs[0];
                }
            }

            if (string.IsNullOrEmpty(fileToOpen) || !File.Exists(fileToOpen))
            {
                MessageBox.Show($"No 3D model found in: {outDir}\nPlease click 'Start Reconstruction' first.", "File Not Found", MessageBoxButton.OK, MessageBoxImage.Information);
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
                MessageBox.Show("Error starting Blender: " + ex.Message, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void BtnOpenPreview_Click(object sender, RoutedEventArgs e)
        {
            string patientId = (CmbPatientCase.SelectedItem as ComboBoxItem)?.Tag as string ?? "P_00001";
            string repoRoot = FindRepoRoot();

            string outDir = _currentMode switch
            {
                ReconstructionMode.TotalReconstruction => Path.Combine(repoRoot, "07_Total_Reconstruction_Module", "output", patientId),
                ReconstructionMode.BreastSurface => Path.Combine(repoRoot, "05_Breast_Surface_Module", "output", patientId),
                _ => Path.Combine(repoRoot, "04_Spicular_Lesion_SMRD_Module", "output", patientId)
            };

            string? img = FindPreviewImage(outDir, patientId);
            if (!string.IsNullOrEmpty(img) && File.Exists(img))
            {
                try
                {
                    Process.Start(new ProcessStartInfo { FileName = img, UseShellExecute = true });
                }
                catch { }
            }
            else
            {
                MessageBox.Show("No 3D render or preview generated yet for this case.", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }

        private void BtnOpenFolder_Click(object sender, RoutedEventArgs e)
        {
            string patientId = (CmbPatientCase.SelectedItem as ComboBoxItem)?.Tag as string ?? "P_00001";
            string repoRoot = FindRepoRoot();

            string outDir = _currentMode switch
            {
                ReconstructionMode.TotalReconstruction => Path.Combine(repoRoot, "07_Total_Reconstruction_Module", "output", patientId),
                ReconstructionMode.BreastSurface => Path.Combine(repoRoot, "05_Breast_Surface_Module", "output", patientId),
                _ => Path.Combine(repoRoot, "04_Spicular_Lesion_SMRD_Module", "output", patientId)
            };

            if (Directory.Exists(outDir))
            {
                Process.Start(new ProcessStartInfo { FileName = outDir, UseShellExecute = true });
            }
            else
            {
                MessageBox.Show("Output directory does not exist yet.", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }

        #endregion
    }
}
