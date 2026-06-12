using HTTP
import JSON
using ZipFile


"""
Unzip an archive to a directory, maintaining the internal folder structure.

Thanks to sylvaticus;
https://discourse.julialang.org/t/how-to-extract-a-file-in-a-zip-archive-without-using-os-specific-tools/34585/5
"""
function unzip(file, dir)
    zarchive = ZipFile.Reader(file)
    for f in zarchive.files
        file_path = joinpath(dir, f.name)
        if (endswith(f.name,"/") || endswith(f.name,"\\"))
            mkdir(file_path)
        else
            write(file_path, read(f))
        end
    end
    close(zarchive)
end

"""
Use a Zenodo DOI to retrieve the URL of the zip file.
"""
function get_zenodo_file_url(doi::String)
    if !occursin(r"^10\.5281\/zenodo.\d*", doi)
        error("Invalid zenodo DOI")
    end
    record = replace(doi, "10.5281/zenodo." => "")
    req = string("https://zenodo.org/api/records/", record)
    response = HTTP.get(req)
    status = response.status
    if status != 200
        error("Unexpected response $status. Zenodo download failed.")
    end
    json_response = JSON.parse(response.body)
    return json_response["files"][1]["key"], json_response["files"][1]["links"]["self"]
end

"""
Download all data of a Zenodo entry, based on its DOI.

For example:
    download_zenodo_files("10.5281/zenodo.20432376", pwd())
Will download and unzip the HTTP.jl archived repository to the current working directory.
"""
function download_zenodo_files(doi::String, output_dir)
    if !isdir(output_dir)
        error("Output directory does not exist.")
    end
    fname, download_url = get_zenodo_file_url(doi)
    response = HTTP.get(download_url)
    status = response.status
    if status != 200
        error("Unexpected response $status. Zenodo download failed.")
    end
    unzip(IOBuffer(response.body), output_dir)
    return nothing
end
