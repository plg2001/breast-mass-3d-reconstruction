# Module 02 - Desktop Client WPF (.NET 8) & Core Unity Library

Clinical desktop application and scientific suite for 3D mammographic reconstruction, morphometric analysis, and VR / Edge integration.

---

## 🌟 Supported Reconstruction Modes

The minimalist GUI allows users to select between **3 reconstruction modes**:

### 1. 🌐 Complete Reconstruction (Breast + Lesion) - Module 07
- **Description**: Generates the complete patient volumetric model composed of **15 continuous biological layers** adhering to the FDA VICTRE standard:
  - Skin, Subcutaneous Adipose Tissue, Deep Pectoral Fascia, Retromammary Space, Pectoralis Major, Rib Cage, Nipple-Areola Complex, Cooper's Ligaments, Glandular Lobes, Lactiferous Ducts, Superficial & Deep Vascular System, Lymph Nodes, Nerves, and Lymphatics.
  - **Spicular 3D SMRD Tumor** biologically embedded in the parenchyma with exact stereotactic localization (clinical quadrant, e.g. UOQ).
- **Customizable Parameters**: Spicularity factor (0.5x - 2.5x), number of 3D radiating spicules (20 - 80).
- **Generated Outputs**: `P_XXXXX_total_3d.glb`, `P_XXXXX_total_scene.blend`, `P_XXXXX_total_preview.png`, clinical report `_total_clinical_report.json`.

### 2. 🍈 Breast Surface & Volumetrics - Module 05
- **Description**: Extracts the continuous bi-planar skin envelope, kinematic PNL (Posterior Nipple Line) alignment, and super-elliptic lofting based on literature (Huang et al., Pinto et al., Rodríguez-Ruiz et al.).
- **Required Inputs**: Only the CC and MLO radiographic views (ROI lesion masks are not required).
- **Extracted Metrics**: Estimated breast volume (cm³), skin surface area (cm²), dimensions W x D x H (mm), watertight 2-manifold check.
- **Generated Outputs**: `P_XXXXX_breast_surface.glb`, `P_XXXXX_scene.blend`, `P_XXXXX_preview.png`, `P_XXXXX_metrics.json`.

### 3. 🎯 3D Lesion Only - Modules 04 and 01
- **Description**: Isolated 3D synthesis of the tumor mass via academic algorithms:
  - **SMRD (2015)** - Luis de Sisternes et al. (Medical Physics): Spherical harmonics radial deformation.
  - **Ling Shao (2007)** - Journal of Digital Imaging: Algebraic visual hull and partial volume compensation (PVC).
- **Extracted Metrics**: Tumor volume (mm³), biological sphericity (Ψ), 2D Dice scores (CC and MLO), mean deviation error.

---

## 📁 Project Structure

```
02_Desktop_Client_WPF/
├── Mammography3D_Solution.sln          # Visual Studio / .NET 8 Solution
├── Mammography3D.Desktop/               # WPF GUI Application
│   ├── MainWindow.xaml                  # Modern Clean Dark Medical Theme
│   ├── MainWindow.xaml.cs               # Event handling, auto-discovery & Blender launcher
│   └── App.xaml
├── Mammography3D.Core/                  # Shared Library (.NET 8 and Unity C#)
│   ├── Models/
│   │   ├── ReconstructionConfig.cs      # Unified configuration for all modes
│   │   ├── ReconstructionResult.cs      # 3D files and clinical metrics
│   │   └── ReconstructionProgress.cs    # Real-time progress and logging
│   ├── Services/
│   │   ├── IReconstructionService.cs    # Service interface
│   │   └── LocalPythonReconstructionService.cs # Python subprocess executor & JSON parser
│   └── Unity/                           # Reusable Unity bridge components
└── Mammography3D_App/                   # Pre-compiled standalone release executable
```

---

## 🚀 Quick Start

Double-click the batch launcher in the repository root:
```powershell
.\Mammography3D.bat
```
or run directly:
```powershell
.\02_Desktop_Client_WPF\Mammography3D_App\Mammography3D.Desktop.exe
```
