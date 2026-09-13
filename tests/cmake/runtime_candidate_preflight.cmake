if(NOT DEFINED CANDIDATE OR NOT DEFINED REPO_ROOT OR NOT DEFINED FIXTURE_ROOT)
    message(FATAL_ERROR "Missing runtime candidate test arguments")
endif()
file(MAKE_DIRECTORY "${FIXTURE_ROOT}/assets")
file(WRITE "${FIXTURE_ROOT}/assets/fixture.engine" "deliberately not an engine")
file(WRITE "${FIXTURE_ROOT}/outside.engine" "deliberately not an engine")
file(SHA256 "${FIXTURE_ROOT}/assets/fixture.engine" fixture_hash)
set(calibration [=[{
  "schema_version": 1, "profile_id": "fixture_profile",
  "calibrated_at": "2026-09-12T00:00:00Z", "resolution_width": 1920,
  "resolution_height": 1080, "fov_horizontal_deg": 103.0,
  "in_game_sensitivity": 1.0, "counts_per_pixel_x": 1.25,
  "counts_per_pixel_y": 1.5, "deadband_counts": 0.0,
  "nonlinearity_alpha": 0.0, "cross_coupling_xy": 0.0,
  "rmse_pixels": 0.5
}]=])
file(WRITE "${FIXTURE_ROOT}/calibration.json" "${calibration}")
set(common "--assets=${FIXTURE_ROOT}/assets" "--calibration=${FIXTURE_ROOT}/calibration.json"
    "--scenario=${REPO_ROOT}/configs/scenario/aimlabs.yaml" --fov=103 --sensitivity=1 --preflight)

function(expect_rejected expected)
    execute_process(COMMAND "${CANDIDATE}" ${common} ${ARGN}
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 10)
    if(NOT result STREQUAL "2" OR NOT error MATCHES "${expected}")
        message(FATAL_ERROR "Expected preflight rejection '${expected}', got ${result}: ${output} ${error}")
    endif()
endfunction()

expect_rejected("SHA-256 verification failed" --engine=fixture.engine
    --sha256=0000000000000000000000000000000000000000000000000000000000000000)
expect_rejected("outside the configured trusted" --engine=../outside.engine "--sha256=${fixture_hash}")
string(REPLACE "\"rmse_pixels\": 0.5" "\"rmse_pixels\": 49.0" rejected_fit "${calibration}")
file(WRITE "${FIXTURE_ROOT}/calibration.json" "${rejected_fit}")
expect_rejected("held-out RMSE" --engine=fixture.engine "--sha256=${fixture_hash}")
