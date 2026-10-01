import RemoteBMI.Server: run_bmi_server
using mGV
run_bmi_server(mGV.Model, "127.0.0.1", 8000)