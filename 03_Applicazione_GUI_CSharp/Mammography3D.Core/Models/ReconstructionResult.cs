using System;

namespace Mammography3D.Core.Models
{
    /// <summary>
    /// Encapsulates the results, output file paths, and quantitative metrics
    /// of the 3D lesion reconstruction.
    /// Directly readable by Unity scripts to instantiate 3D objects and show medical metrics.
    /// </summary>
    public class ReconstructionResult
    {
        public bool Success { get; set; }
        public string? ErrorMessage { get; set; }
        public double ExecutionTimeSeconds { get; set; }

        public string OutputDirectory { get; set; } = string.Empty;
        public string? GlbLesionPath { get; set; }
        public string? BlendLesionPath { get; set; }
        public string? ReportJsonPath { get; set; }
        public string? ReportMdPath { get; set; }
        public string? MultiviewRenderPath { get; set; }
        public string? AnalysisPlotPath { get; set; }
        public string? SegmentationPlotPath { get; set; }

        // Scientific Quantitative Metrics (Table 1 Ling Shao 2007)
        public double DiceCc { get; set; }
        public double IoUCc { get; set; }
        public double DiceMlo { get; set; }
        public double IoUMlo { get; set; }

        public double InitialMeanDevRatePercent { get; set; }
        public double ArtMeanDevRatePercent { get; set; }
        public double FinalPvcMeanDevRatePercent { get; set; }
        public double PaperBenchmarkPercent { get; set; } = 5.82;

        // 3D Geometric Properties
        public int VerticesCount { get; set; }
        public int FacesCount { get; set; }
        public double VolumeMm3 { get; set; }
        public double SurfaceAreaMm2 { get; set; }
        public double Sphericity { get; set; }
        public double[] WorldCentroid { get; set; } = new double[3];

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
