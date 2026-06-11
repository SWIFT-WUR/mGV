using HTTP
import JSON


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
    write(
        joinpath(output_dir, basename(fname)),
        response.body
    )
    return nothing
end
