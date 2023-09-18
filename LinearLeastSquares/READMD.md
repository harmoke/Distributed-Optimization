Just run main.py to see the results.  
You may need to check and change the path in main.py file.  
Change $m$ and $n$ as you want, but be careful with the buffersize and memory limit.  
Here is a way to modify the core parameter:
```
sudo sysctl -w net.inet.udp.maxdgram=65535
sudo sysctl -w net.core.rmem_max=4194304
```
