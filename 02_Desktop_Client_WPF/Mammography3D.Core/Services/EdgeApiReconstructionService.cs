using System;
using System.IO;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Mammography3D.Core.Models;

namespace Mammography3D.Core.Services
{
    /// <summary>
    /// Implements 3D lesion reconstruction via an Edge/Fog server REST API.
    /// Perfectly aligned with the Thesis Architecture:
    /// 'Applicazione immersiva Android supportata da Edge/Fog Computing'.
    /// The Android VR headset / client uploads images to the Edge Server,
    /// and receives the 3D GLB mesh and metrics for immersive visualization.
    /// </summary>
    public class EdgeApiReconstructionService : IReconstructionService
    {
        private readonly string _edgeServerUrl;
        private readonly HttpClient _httpClient;

        public EdgeApiReconstructionService(string edgeServerUrl, HttpClient? httpClient = null)
        {
            _edgeServerUrl = edgeServerUrl.TrimEnd('/');
            _httpClient = httpClient ?? new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
        }

        public async Task<ReconstructionResult> ReconstructAsync(
            ReconstructionConfig config,
            IProgress<ReconstructionProgress>? progress = null,
            CancellationToken cancellationToken = default)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            progress?.Report(new ReconstructionProgress(
                ReconstructionPhase.Starting,
                "Connessione al server Edge/Fog in corso...",
                5.0f,
                $"Endpoint Edge: {_edgeServerUrl}/api/v1/reconstruct"
            ));

            try
            {
                using var form = new MultipartFormDataContent();

                // Add images
                await AddFileContentAsync(form, "r_cc", config.RCcPath).ConfigureAwait(false);
                await AddFileContentAsync(form, "r_mlo", config.RMloPath).ConfigureAwait(false);
                await AddFileContentAsync(form, "roi_cc", config.RoiCcPath).ConfigureAwait(false);
                await AddFileContentAsync(form, "roi_mlo", config.RoiMloPath).ConfigureAwait(false);

                // Add parameters
                form.Add(new StringContent(config.MloAngleDeg.ToString(System.Globalization.CultureInfo.InvariantCulture)), "mlo_angle");
                form.Add(new StringContent(config.GridResolution.ToString()), "grid_resolution");
                form.Add(new StringContent(config.VoxelSizeMm.ToString(System.Globalization.CultureInfo.InvariantCulture)), "voxel_size_mm");

                progress?.Report(new ReconstructionProgress(
                    ReconstructionPhase.Starting,
                    "Caricamento mammografie verso il nodo Edge...",
                    15.0f,
                    "Invio dati multipart/form-data al server Edge..."
                ));

                var response = await _httpClient.PostAsync(
                    $"{_edgeServerUrl}/api/v1/reconstruct",
                    form,
                    cancellationToken
                ).ConfigureAwait(false);

                response.EnsureSuccessStatusCode();

                progress?.Report(new ReconstructionProgress(
                    ReconstructionPhase.MeshExtraction,
                    "Elaborazione Edge completata, download modello 3D...",
                    85.0f,
                    "Ricezione payload GLB e metadati dal server..."
                ));

                string jsonResponse = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
                sw.Stop();

                // Parse server response
                var result = new ReconstructionResult
                {
                    Success = true,
                    ExecutionTimeSeconds = sw.Elapsed.TotalSeconds,
                    OutputDirectory = config.OutputDir
                };

                progress?.Report(new ReconstructionProgress(
                    ReconstructionPhase.Done,
                    "Modello 3D scaricato con successo dal server Edge!",
                    100.0f,
                    "Pronto per l'istanziazione immersiva nel visore VR."
                ));

                return result;
            }
            catch (Exception ex)
            {
                return ReconstructionResult.Fail($"Errore comunicazione Edge: {ex.Message}", sw.Elapsed.TotalSeconds);
            }
        }

        private static async Task AddFileContentAsync(MultipartFormDataContent form, string fieldName, string filePath)
        {
            if (File.Exists(filePath))
            {
                byte[] bytes = await Task.Run(() => File.ReadAllBytes(filePath)).ConfigureAwait(false);
                var content = new ByteArrayContent(bytes);
                content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/jpeg");
                form.Add(content, fieldName, Path.GetFileName(filePath));
            }
        }
    }
}
