import RemoteBMI.Server: run_bmi_server
using mGV
run_bmi_server(mGV.Model, "localhost", 8000)