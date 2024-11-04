IMAGE_SIZE = 8
KERNEL_SIZE = 2
CHANNELS_IN = 1
CHANNELS_OUT = 10
STRIDE = 1
PADDING = 0

def clock_cycles_partial():
    return ((IMAGE_SIZE + 2*PADDING - KERNEL_SIZE)/STRIDE+1)**2 * CHANNELS_OUT

def adders_partial():
    return KERNEL_SIZE**2-1

def multipliers_partial():
    return KERNEL_SIZE**2

#####

def clock_cycles_none():
    return 1

def adders_none():
    return (KERNEL_SIZE**2-1) * CHANNELS_IN * CHANNELS_OUT

def multipliers_none():
    return KERNEL_SIZE**2 * CHANNELS_IN * CHANNELS_OUT

def clock_cycles_full():
    return KERNEL_SIZE**2 * CHANNELS_IN * ((IMAGE_SIZE + 2*PADDING - KERNEL_SIZE)/STRIDE+1)**2 * CHANNELS_OUT

def adders_full():
    return 1

def multipliers_full():
    return 1

print(clock_cycles_partial())
print(adders_partial())
print(multipliers_partial())

print(clock_cycles_none())
print(adders_none())
print(multipliers_none())

print(clock_cycles_full())
print(adders_full())
print(multipliers_full())
