fx_version "cerulean"
game "gta5"

author "Wormhole Studios"
description "worm_instruments_assets"
version "0.1.0"

-- worm_instruments' props, clips and effects, apart from the script: restarting the script leaves them loaded. Never
-- restart this resource while the server runs; after an update, restart the whole server (the game reads props and
-- clips only when it starts).
--
-- The props' archetype files (stream/*.ytyp) are read through one DLC_ITYP_REQUEST glob: every prop's .ytyp, from the
-- first one (the harmonica, Phase 3), is picked up by it.
files {
    "stream/*.ytyp"
}

-- The script's own props (Source/ keeps how each was made).
data_file "DLC_ITYP_REQUEST" "stream/*.ytyp"
