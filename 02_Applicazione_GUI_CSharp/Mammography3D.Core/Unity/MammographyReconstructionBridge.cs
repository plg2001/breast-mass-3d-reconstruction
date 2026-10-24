#if UNITY_5_3_OR_NEWER || UNITY_2017_1_OR_NEWER
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Events;
using Mammography3D.Core.Models;
using Mammography3D.Core.Services;

namespace Mammography3D.Unity
{
    /// <summary>
    /// Unity MonoBehaviour Bridge for 3D Mammographic Lesion Reconstruction.
    /// Drop this component onto a GameObject in your Unity Scene.
    /// Supports both:
    /// 1. Local execution (in Unity Editor / PC Build) via Python subprocess.
    /// 2. Remote Edge/Fog execution (on Android VR Headsets like Meta Quest / Pico) via HTTP REST API.
    /// </summary>
    public class MammographyReconstructionBridge : MonoBehaviour
    {
        [Header("Backend Mode")]
        [Tooltip("True to use remote Edge/Fog Server (recommended for Android VR visor). False for local Python execution.")]
        public bool useEdgeServer = false;

        [Tooltip("URL of the Edge/Fog Server (e.g. http://192.168.1.50:8000)")]
        public string edgeServerUrl = "http://127.0.0.1:8000";

        [Header("Input Image Paths (StreamingAssets or PersistentDataPath)")]
        public string rCcImagePath = "01_Ricostruzione_Lesione_3D/data/R-CC.jpg";
        public string rMloImagePath = "01_Ricostruzione_Lesione_3D/data/R-MLO.jpg";
        public string roiCcImagePath = "01_Ricostruzione_Lesione_3D/data/ROI-R-CC.jpg";
        public string roiMloImagePath = "01_Ricostruzione_Lesione_3D/data/ROI-R-MLO.jpg";

        [Header("Reconstruction Parameters (Ling Shao 2007)")]
        [Range(15f, 75f)]
        public float mloAngle = 45f;

        [Range(50, 250)]
        public int gridResolution = 140;

        public float voxelSizeMm = 0.1f;

        [Header("Scene Spawn Anchor")]
        [Tooltip("Transform where the reconstructed 3D lesion will be instantiated")]
        public Transform lesionSpawnAnchor;

        [Header("Unity Events")]
        public UnityEvent<float, string> OnProgressUpdated;
        public UnityEvent<ReconstructionResult> OnReconstructionSuccess;
        public UnityEvent<string> OnReconstructionFailed;

        private CancellationTokenSource _cts;

        /// <summary>
        /// Starts the 3D reconstruction asynchronously without freezing Unity's main thread.
        /// </summary>
        public async void StartReconstruction()
        {
            _cts?.Cancel();
            _cts = new CancellationTokenSource();

            var config = new ReconstructionConfig
            {
                RCcPath = ResolvePath(rCcImagePath),
                RMloPath = ResolvePath(rMloImagePath),
                RoiCcPath = ResolvePath(roiCcImagePath),
                RoiMloPath = ResolvePath(roiMloImagePath),
                MloAngleDeg = mloAngle,
                GridResolution = gridResolution,
                VoxelSizeMm = voxelSizeMm,
                OutputDir = Path.Combine(Application.persistentDataPath, "Mammography3D_Output")
            };

            IReconstructionService service = useEdgeServer
                ? new EdgeApiReconstructionService(edgeServerUrl)
                : new LocalPythonReconstructionService();

            var progress = new Progress<ReconstructionProgress>(p =>
            {
                OnProgressUpdated?.Invoke(p.Percentage, p.StageDescription);
                Debug.Log($"[Mammography3D] {p.Percentage:F1}% - {p.StageDescription}");
            });

            try
            {
                Debug.Log("<color=cyan>[Mammography3D] Starting 3D lesion reconstruction...</color>");
                var result = await service.ReconstructAsync(config, progress, _cts.Token);

                if (result.Success)
                {
                    Debug.Log($"<color=green>[Mammography3D] 3D Model created in {result.ExecutionTimeSeconds:F2}s!</color>");
                    OnReconstructionSuccess?.Invoke(result);

                    // Load .glb model into Unity scene
                    if (!string.IsNullOrEmpty(result.GlbLesionPath) && File.Exists(result.GlbLesionPath))
                    {
                        LoadGlbIntoScene(result.GlbLesionPath, result);
                    }
                }
                else
                {
                    Debug.LogError($"[Mammography3D] Reconstruction failed: {result.ErrorMessage}");
                    OnReconstructionFailed?.Invoke(result.ErrorMessage ?? "Unknown error");
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Mammography3D] Exception during reconstruction: {ex.Message}");
                OnReconstructionFailed?.Invoke(ex.Message);
            }
        }

        public void CancelReconstruction()
        {
            _cts?.Cancel();
            Debug.Log("[Mammography3D] Reconstruction cancelled by user.");
        }

        private string ResolvePath(string path)
        {
            if (Path.IsPathRooted(path) && File.Exists(path))
                return path;

            string streamingPath = Path.Combine(Application.streamingAssetsPath, path);
            if (File.Exists(streamingPath))
                return streamingPath;

            string dataPath = Path.Combine(Application.dataPath, "..", path);
            if (File.Exists(dataPath))
                return Path.GetFullPath(dataPath);

            return path;
        }

        private void LoadGlbIntoScene(string glbPath, ReconstructionResult result)
        {
            Debug.Log($"[Mammography3D] Loading GLB model into Unity scene: {glbPath}");
            Transform parentAnchor = lesionSpawnAnchor != null ? lesionSpawnAnchor : transform;

            // Reflection-based instantiation so it compiles out-of-the-box before glTFast is installed
            var gltfType = Type.GetType("GLTFast.GltfAsset, glTFast");
            if (gltfType != null)
            {
                var gltfGo = new GameObject("Reconstructed_Lesion_3D");
                gltfGo.transform.SetParent(parentAnchor, false);
                var component = gltfGo.AddComponent(gltfType);
                var loadMethod = gltfType.GetMethod("Load", new[] { typeof(string) });
                loadMethod?.Invoke(component, new object[] { $"file://{Path.GetFullPath(glbPath)}" });
                Debug.Log("<color=green>[Mammography3D] 3D Lesion model instantiated via glTFast runtime loader!</color>");

                // Attach VR annotation marker if present
                var marker = gltfGo.GetComponentInChildren<LesionAnnotationMarker>();
                if (marker != null)
                {
                    marker.PopulateWithMetrics(result);
                }
            }
            else
            {
                Debug.LogWarning("[Mammography3D] glTFast non rilevato. Per visualizzare il .glb a runtime nel visore VR, installa 'glTFast' dal Package Manager di Unity (com.atteneder.gltfast).");
            }
        }

        private void OnDestroy()
        {
            _cts?.Cancel();
        }
    }
}
#endif
