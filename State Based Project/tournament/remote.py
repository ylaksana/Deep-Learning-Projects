from . import runner, utils

# TODO: Wrap TeamRunner and Team in ray if possible
try:
    import ray

    #@ray.remote(num_gpus=1)
    @ray.remote
    class RayMatch(runner.Match):
        pass

    #@ray.remote(num_gpus=1)
    @ray.remote
    class RayTeamRunner(runner.TeamRunner):
        pass

    #@ray.remote(num_gpus=1)
    @ray.remote
    class RayStateRecorder(utils.StateRecorder):
        pass

    #@ray.remote(num_gpus=1)
    @ray.remote
    class RayDataRecorder(utils.DataRecorder):
        pass

    #@ray.remote(num_gpus=1)
    @ray.remote
    class RayVideoRecorder(utils.VideoRecorder):
        pass

    RayMatchException = ray.exceptions.RayTaskError

    get = ray.get
    init = ray.init

except ImportError:
    ray = None
