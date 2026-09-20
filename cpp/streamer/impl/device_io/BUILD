# Getting bytes onto a device, once they have been read into host memory.
#
# The sibling of async_io/, which moves them from storage into host memory. Splitting them this way
# keeps the device path out of the packages that serve host destinations: everything here is reached
# only when a submission names a device, and nothing here is built or started otherwise.
#
# The sub-packages restrict their own visibility to //streamer. This file holds no targets - it is
# here so the grouping has somewhere to be explained.
