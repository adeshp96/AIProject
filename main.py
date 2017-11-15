# This is an example of popping a packet from the Emotiv class's packet queue
# Additionally, exports the data to a CSV file.
# You don't actually need to dequeue packets, although after some time allocation lag may occur if you don't.


import platform
import time
from mlp import setting
import mlp

from emokit.emotiv import Emotiv

if platform.system() == "Windows":
    pass

# file = 'dataset/'+ setting + '/adesh/left.csv'
# file = 'emotiv_values_2017-10-12 14-25-56.771261.csv'
file = 'emotiv_encrypted_data_UD20160103001874_2017-04-05.17-21-32.384061.csv'
# print "Reading from file",file

if __name__ == "__main__":
    with Emotiv(display_output=True, write = True) as headset:
    # with Emotiv(display_output=False, verbose=True, write = True) as headset:
        print("Serial Number: %s" % headset.serial_number)

        while headset.running:
            try:
                packet = headset.dequeue()
            except Exception:
                headset.stop()
                print "Error"
            time.sleep(0.001)
