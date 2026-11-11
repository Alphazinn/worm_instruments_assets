fx_version "cerulean"
game "gta5"

author "Wormhole Studios"
description "worm_instruments_assets"
version "0.1.0"

-- worm_instruments' props, clips, effects and native sounds, apart from the script: restarting the script leaves them
-- loaded. Never restart this resource while the server runs; after an update, restart the whole server (the game reads
-- props, clips and sound data only when it starts).
--
-- The native sound data (from Phase 5) is named winstruments_sounds.dat54.rel: the game keys a sound data file by its
-- name up to the first "_", and "worm_" is already another resource's (CODESTYLE 2.25).
--
-- The props' archetype files (stream/*.ytyp) are read through one DLC_ITYP_REQUEST glob: every prop's .ytyp, from the
-- first one (the harmonica, Phase 3), is picked up by it.
files {
    "stream/*.ytyp"
}

-- The script's own props (Source/ keeps how each was made).
data_file "DLC_ITYP_REQUEST" "stream/*.ytyp"
