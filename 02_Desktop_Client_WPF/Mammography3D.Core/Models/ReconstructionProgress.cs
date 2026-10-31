namespace Mammography3D.Core.Models
{
    public enum ReconstructionPhase
    {
        Starting,
        SegmentationCC,
        SegmentationMLO,
        VisualHull,
        ART,
        PartialVolumeEstimation,
        MeshExtraction,
        ExportingGLB,
        Done,
        Failed,
        Cancelled
    }

    /// <summary>
    /// Represents progress update during 3D reconstruction.
    /// Dispatched via IProgress&lt;ReconstructionProgress&gt; to WPF UI or Unity Coroutine/Task.
    /// </summary>
    public class ReconstructionProgress
    {
        public ReconstructionPhase Phase { get; set; }
        public string StageDescription { get; set; } = string.Empty;
        public float Percentage { get; set; }
        public string LogMessage { get; set; } = string.Empty;
        public bool IsError { get; set; }

        public ReconstructionProgress(ReconstructionPhase phase, string stageDescription, float percentage, string logMessage, bool isError = false)
        {
            Phase = phase;
            StageDescription = stageDescription;
            Percentage = percentage;
            LogMessage = logMessage;
            IsError = isError;
        }
    }
}
