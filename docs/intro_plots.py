import pathlib
import etrack as et
import matplotlib.pyplot as plt

dataset = pathlib.Path.cwd().joinpath(pathlib.PosixPath("test/2022lepto01_converted_2024.03.27_0.mp4.nix"))
ntd = et.NixtrackData(str(dataset))
td = ntd.track_data(bodypart="snout")

x, y, time, score = td.positions()

plt.scatter(x, y, c=score)
plt.xlabel("x-position")
plt.ylabel("y-position")
plt.savefig(str(pathlib.Path.cwd().joinpath("docs/intro_scatter.png")))
plt.close()

print(td.quality_threshold)
td.quality_threshold = 0.8
print(td.position_limits)
td.position_limits = (400, 50, 3100, 450)
td.filter_tracks()

x, y, time, score = td.positions()

plt.scatter(x, y, c=time)
plt.xlabel("x-position")
plt.ylabel("y-position")
plt.savefig(str(pathlib.Path.cwd().joinpath("docs/intro_scatter2.png")))
plt.close()