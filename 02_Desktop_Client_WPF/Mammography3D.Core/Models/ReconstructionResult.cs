using System;

namespace Mammography3D.Core.Models
{
    /// <summary>
    /// Incapsula i risultati dell'elaborazione 3D, i file esportati (.glb, .blend, preview)
    /// e le metriche quantitative clinico-scientifiche per tutte le modalità della suite.
    /// </summary>
    public class ReconstructionResult
    {
        public bool Success { get; set; }
        public string? ErrorMessage { get; set; }
        public double ExecutionTimeSeconds { get; set; }
        public ReconstructionMode Mode { get; set; } = ReconstructionMode.TotalReconstruction;
        public string PatientId { get; set; } = string.Empty;

        // Cartelle e file generati
        public string OutputDirectory { get; set; } = string.Empty;
        public string? GlbPath { get; set; }
        public string? BlendPath { get; set; }
        public string? PreviewImagePath { get; set; }
        public string? JsonReportPath { get; set; }
        public string? MarkdownReportPath { get; set; }

        // Alias retrocompatibili
        public string? GlbLesionPath { get => GlbPath; set => GlbPath = value; }
        public string? BlendLesionPath { get => BlendPath; set => BlendPath = value; }
        public string? ReportJsonPath { get => JsonReportPath; set => JsonReportPath = value; }
        public string? ReportMdPath { get => MarkdownReportPath; set => MarkdownReportPath = value; }
        public string? MultiviewRenderPath { get => PreviewImagePath; set => PreviewImagePath = value; }
        public string? AnalysisPlotPath { get; set; }
        public string? SegmentationPlotPath { get; set; }

        // Metriche Modulo 07: Ricostruzione Totale
        public string? Pathology { get; set; }
        public double AiConfidence { get; set; }
        public string? TumorQuadrant { get; set; }
        public string? TumorQuadrantName { get; set; }
        public string? TumorLaterality { get; set; }
        public double TumorDiameterMm { get; set; }
        public double SpicularityFactor { get; set; }
        public int Total3DSpikes { get; set; }
        public int BiologicalLayersCount { get; set; } = 15;

        // Metriche Morfometriche Seno (Moduli 05 e 07)
        public double BreastVolumeCm3 { get; set; }
        public double BreastSurfaceAreaCm2 { get; set; }
        public double BreastWidthMm { get; set; }
        public double BreastDepthPnlMm { get; set; }
        public double BreastHeightMm { get; set; }
        public bool IsWatertight { get; set; } = true;

        // Metriche Lesione Tumorale (Moduli 01, 04 e 07)
        public double VolumeMm3 { get; set; }
        public double Sphericity { get; set; }
        public int VerticesCount { get; set; }
        public int FacesCount { get; set; }
        public double SurfaceAreaMm2 { get; set; }
        public double[] WorldCentroid { get; set; } = new double[3];

        // Benchmark e Confronti Clinici 2D (Dice Score, PVE, Deviation Rate)
        public double DiceCc { get; set; }
        public double IoUCc { get; set; }
        public double DiceMlo { get; set; }
        public double IoUMlo { get; set; }
        public double InitialMeanDevRatePercent { get; set; }
        public double ArtMeanDevRatePercent { get; set; }
        public double FinalPvcMeanDevRatePercent { get; set; }
        public double PaperBenchmarkPercent { get; set; } = 5.82;

        public static ReconstructionResult Fail(string error, double executionTime = 0)
        {
            return new ReconstructionResult
            {
                Success = false,
                ErrorMessage = error,
                ExecutionTimeSeconds = executionTime
            };
        }
    }
}
