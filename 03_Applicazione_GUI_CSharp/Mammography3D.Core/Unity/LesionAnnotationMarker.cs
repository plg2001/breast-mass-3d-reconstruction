#if UNITY_5_3_OR_NEWER || UNITY_2017_1_OR_NEWER
using UnityEngine;
using UnityEngine.UI;
using Mammography3D.Core.Models;

namespace Mammography3D.Unity
{
    /// <summary>
    /// Interactive 3D Marker and Annotation Panel for the Lesion in Unity VR.
    /// Directly implements Thesis Proposal Steps 4, 5, 6:
    /// 'L'utente guarda/seleziona il marker della lesione tramite il visore...
    /// si apre un pannello 3D con identificativo, volume, sfericità, caratteristiche'.
    /// </summary>
    public class LesionAnnotationMarker : MonoBehaviour
    {
        [Header("VR UI Panel Reference")]
        public GameObject detailsPanel;
        public Text titleText;
        public Text metricsText;
        public Text sphericityText;
        public Text volumeText;

        [Header("Visual Marker Settings")]
        public Transform billboardAnchor;
        public Renderer markerPulsingRenderer;
        public Color warningColor = new Color(0.9f, 0.2f, 0.2f, 0.9f);

        private bool _isPanelVisible = false;

        private void Start()
        {
            if (detailsPanel != null)
                detailsPanel.SetActive(false);
        }

        private void Update()
        {
            // Billboard effect: Make annotation always face the VR headset camera
            if (billboardAnchor != null && Camera.main != null)
            {
                billboardAnchor.LookAt(Camera.main.transform);
                billboardAnchor.Rotate(0, 180, 0); // Face camera
            }
        }

        /// <summary>
        /// Populates the annotation marker with scientific data from the Ling Shao (2007) reconstruction.
        /// </summary>
        public void PopulateWithMetrics(ReconstructionResult result)
        {
            if (titleText != null)
                titleText.text = "Lesione Ricostruita 3D (Ling Shao 2007)";

            if (volumeText != null)
                volumeText.text = $"Volume 3D: {result.VolumeMm3:F2} mm³";

            if (sphericityText != null)
                sphericityText.text = $"Sfericità (Ψ): {result.Sphericity:F3}";

            if (metricsText != null)
            {
                metricsText.text =
                    $"Mean Dev Rate: {result.FinalPvcMeanDevRatePercent:F2}%\n" +
                    $"Dice Score CC: {result.DiceCc * 100:F1}%\n" +
                    $"Dice Score MLO: {result.DiceMlo * 100:F1}%\n" +
                    $"Mesh: {result.FacesCount:N0} facce ({result.VerticesCount:N0} vertici)";
            }
        }

        /// <summary>
        /// Called when the VR user gazes at or pinches/clicks the 3D lesion marker.
        /// </summary>
        public void ToggleDetailsPanel()
        {
            _isPanelVisible = !_isPanelVisible;
            if (detailsPanel != null)
                detailsPanel.SetActive(_isPanelVisible);
        }
    }
}
#endif
