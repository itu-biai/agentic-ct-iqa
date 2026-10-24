import os, time
# Keep the kernel busy while files are uploaded through the contents API (an idle kernel lets Colab reclaim the VM).
# Returns when /content/hold_stop exists or after 3 h. Prints HOLD_DONE.
start = time.time()
while not os.path.exists("/content/hold_stop") and time.time() - start < 3 * 3600:
    time.sleep(30)
print("HOLD_DONE", round((time.time() - start) / 60, 1), "min", flush=True)
