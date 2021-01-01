import GPUtil

class GPUUsage:
  def __init__(self):
    self.prev = 0

  def printm(self):
    GPUs = GPUtil.getGPUs()
    gpu = GPUs[0]
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Delta: {2:.0f}MB | Util {3:3.0f}% | Total     {4:.0f}MB".format(
      gpu.memoryFree,
      gpu.memoryUsed,
      gpu.memoryUsed - self.prev,
      gpu.memoryUtil*100,
      gpu.memoryTotal
    ))
    self.prev = gpu.memoryUsed
    return gpu.memoryUtil*100

gpu_usage = GPUUsage()
