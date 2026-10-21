module MetalExt

using Metal: Metal, MetalBackend
import mGV

function __init__()
    return mGV.register_backend!("metal", "Metal", Metal.functional, MetalBackend)
end

end
