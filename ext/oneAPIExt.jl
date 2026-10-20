module oneAPIExt

using oneAPI: oneAPI, oneAPIBackend
import mGV

function __init__()
    mGV.register_backend!("oneapi", "oneAPI", oneAPI.functional, oneAPIBackend)
end

end
