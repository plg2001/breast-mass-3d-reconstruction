using System;
using System.Collections.Generic;
using System.IO;

namespace Mammography3D.Core.Models
{
    /// <summary>
    /// Configuration parameters for the 3D mammographic lesion reconstruction pipeline.
    /// Compatible with Unity (.NET Standard 2.0 / .NET Standard 2.1).
    /// </summary>
    public class ReconstructionConfig
    {
        public string RCcPath { get; set; } = string.Empty;
        public string RMloPath { get; set; } = string.Empty;
        public string RoiCcPath { get; set; } = string.Empty;
        public string RoiMloPath { get; set; } = string.Empty;
        public string OutputDir { get; set; } = string.Empty;

        /// <summary>
        /// MLO projection angle in degrees from horizontal (standard clinical: 45.0°).
        /// </summary>
        public double MloAngleDeg { get; set; } = 45.0;

        /// <summary>
        /// Voxel grid resolution along each axis (default: 140).
        /// </summary>
        public int GridResolution { get; set; } = 140;

        /// <summary>
        /// Voxel size in millimeters (default: 0.1 mm = 100 µm).
        /// </summary>
        public double VoxelSizeMm { get; set; } = 0.1;

        /// <summary>
        /// Optional custom path to Python executable.
        /// If empty, the service will locate Python automatically.
        /// </summary>
        public string? PythonExecutable { get; set; }

        /// <summary>
        /// Optional custom path to lesion_pvc_reconstruction.py.
        /// If empty, the service will locate it in Test Codice or workspace root.
        /// </summary>
        public string? ScriptPath { get; set; }

        /// <summary>
        /// Validates that all required input files exist.
        /// </summary>
        public List<string> Validate()
        {
            var errors = new List<string>();

            if (string.IsNullOrWhiteSpace(RCcPath))
                errors.Add("Il percorso dell'immagine R-CC non è specificato.");
            else if (!File.Exists(RCcPath))
                errors.Add($"Immagine R-CC non trovata: {RCcPath}");

            if (string.IsNullOrWhiteSpace(RMloPath))
                errors.Add("Il percorso dell'immagine R-MLO non è specificato.");
            else if (!File.Exists(RMloPath))
                errors.Add($"Immagine R-MLO non trovata: {RMloPath}");

            if (string.IsNullOrWhiteSpace(RoiCcPath))
                errors.Add("Il percorso della maschera ROI CC non è specificato.");
            else if (!File.Exists(RoiCcPath))
                errors.Add($"Maschera ROI CC non trovata: {RoiCcPath}");

            if (string.IsNullOrWhiteSpace(RoiMloPath))
                errors.Add("Il percorso della maschera ROI MLO non è specificato.");
            else if (!File.Exists(RoiMloPath))
                errors.Add($"Maschera ROI MLO non trovata: {RoiMloPath}");

            if (GridResolution < 30 || GridResolution > 500)
                errors.Add("La risoluzione della griglia voxel deve essere compresa tra 30 e 500.");

            if (MloAngleDeg < 10.0 || MloAngleDeg > 80.0)
                errors.Add("L'angolo MLO deve essere compreso tra 10° e 80°.");

            return errors;
        }
    }
}
