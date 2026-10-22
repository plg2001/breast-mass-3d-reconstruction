using System;
using System.Threading;
using System.Threading.Tasks;
using Mammography3D.Core.Models;

namespace Mammography3D.Core.Services
{
    /// <summary>
    /// Service contract for executing 3D mammographic lesion reconstructions.
    /// Can be implemented locally via Python subprocess, or remotely via Edge/Fog HTTP API.
    /// </summary>
    public interface IReconstructionService
    {
        Task<ReconstructionResult> ReconstructAsync(
            ReconstructionConfig config,
            IProgress<ReconstructionProgress>? progress = null,
            CancellationToken cancellationToken = default
        );
    }
}
