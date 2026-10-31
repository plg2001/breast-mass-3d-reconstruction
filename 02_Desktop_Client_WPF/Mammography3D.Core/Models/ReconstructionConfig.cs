using System;
using System.Collections.Generic;
using System.IO;

namespace Mammography3D.Core.Models
{
    /// <summary>
    /// Modalità operative della suite di ricostruzione 3D mammografica.
    /// </summary>
    public enum ReconstructionMode
    {
        /// <summary>
        /// Modulo 07: Ricostruzione Totale Mammella + Lesione SMRD Spicolata (15 strati biologici completi).
        /// </summary>
        TotalReconstruction,

        /// <summary>
        /// Modulo 05: Ricostruzione 3D Superficie Cutanea e Volumetria Seno (Huang / Pinto / Kita).
        /// </summary>
        BreastSurface,

        /// <summary>
        /// Modulo 04: Ricostruzione Lesione Spicolata 3D SMRD (De Sisternes et al. 2015).
        /// </summary>
        LesionSMRD,

        /// <summary>
        /// Modulo 01: Ricostruzione Lesione Analitica Partial Volume Compensated (Ling Shao 2007).
        /// </summary>
        LesionLingShao
    }

    /// <summary>
    /// Compatibilità legacy per l'algoritmo di ricostruzione lesione.
    /// </summary>
    public enum ReconstructionAlgorithm
    {
        SMRD_2015,
        LingShao_2007
    }

    /// <summary>
    /// Parametri di configurazione per l'intera suite di ricostruzione 3D mammografica.
    /// Compatibile con WPF (.NET 8) e Unity (.NET Standard 2.0 / 2.1).
    /// </summary>
    public class ReconstructionConfig
    {
        /// <summary>
        /// Modalità di ricostruzione attiva selezionata dall'utente.
        /// </summary>
        public ReconstructionMode Mode { get; set; } = ReconstructionMode.TotalReconstruction;

        /// <summary>
        /// ID del paziente (es. P_00001, P_00081, P_00005, P_00232).
        /// </summary>
        public string PatientId { get; set; } = "P_00001";

        /// <summary>
        /// Compatibilità con il vecchio selettore di algoritmo lesione.
        /// </summary>
        public ReconstructionAlgorithm Algorithm
        {
            get => Mode == ReconstructionMode.LesionLingShao ? ReconstructionAlgorithm.LingShao_2007 : ReconstructionAlgorithm.SMRD_2015;
            set => Mode = (value == ReconstructionAlgorithm.LingShao_2007) ? ReconstructionMode.LesionLingShao : ReconstructionMode.LesionSMRD;
        }

        // Percorsi Immagini Mammografiche e Maschere Cliniche
        public string RCcPath { get; set; } = string.Empty;
        public string RMloPath { get; set; } = string.Empty;
        public string RoiCcPath { get; set; } = string.Empty;
        public string RoiMloPath { get; set; } = string.Empty;
        public string OutputDir { get; set; } = string.Empty;

        // Parametri Modulo 07: Ricostruzione Totale
        public double Spicularity { get; set; } = 1.25;
        public int NumSpikes { get; set; } = 48;

        // Parametri Modulo 05: Superficie Seno
        public double PixelSpacingMm { get; set; } = 0.05;

        // Parametri Moduli 01 e 04: Sola Lesione
        public double MloAngleDeg { get; set; } = 45.0;
        public int GridResolution { get; set; } = 140;
        public double VoxelSizeMm { get; set; } = 0.1;

        // Eseguibili opzionali
        public string? PythonExecutable { get; set; }
        public string? ScriptPath { get; set; }
        public string? BlenderExecutable { get; set; }

        /// <summary>
        /// Validates input files and parameters for the active reconstruction mode.
        /// </summary>
        public List<string> Validate()
        {
            var errors = new List<string>();

            switch (Mode)
            {
                case ReconstructionMode.TotalReconstruction:
                    if (string.IsNullOrWhiteSpace(PatientId))
                        errors.Add("Please specify a valid Patient ID (e.g. P_00001).");
                    if (Spicularity < 0.2 || Spicularity > 3.0)
                        errors.Add("Spicularity factor must be between 0.2 and 3.0.");
                    if (NumSpikes < 10 || NumSpikes > 150)
                        errors.Add("Number of 3D spicules must be between 10 and 150.");
                    break;

                case ReconstructionMode.BreastSurface:
                    if (string.IsNullOrWhiteSpace(RCcPath) || !File.Exists(RCcPath))
                        errors.Add($"CC mammogram view not found: {RCcPath}");
                    if (string.IsNullOrWhiteSpace(RMloPath) || !File.Exists(RMloPath))
                        errors.Add($"MLO mammogram view not found: {RMloPath}");
                    if (PixelSpacingMm <= 0.001 || PixelSpacingMm > 1.0)
                        errors.Add("Detector pixel spacing must be between 0.01 and 1.0 mm.");
                    break;

                case ReconstructionMode.LesionSMRD:
                case ReconstructionMode.LesionLingShao:
                    if (string.IsNullOrWhiteSpace(RoiCcPath) || !File.Exists(RoiCcPath))
                        errors.Add($"ROI CC mask not found: {RoiCcPath}");
                    if (string.IsNullOrWhiteSpace(RoiMloPath) || !File.Exists(RoiMloPath))
                        errors.Add($"ROI MLO mask not found: {RoiMloPath}");
                    if (GridResolution < 30 || GridResolution > 500)
                        errors.Add("Voxel grid resolution must be between 30 and 500.");
                    if (MloAngleDeg < 10.0 || MloAngleDeg > 80.0)
                        errors.Add("MLO angle must be between 10° and 80°.");
                    break;
            }

            return errors;
        }
    }
}
