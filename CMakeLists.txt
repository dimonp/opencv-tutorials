cmake_minimum_required(VERSION 3.15)
project(OpenCV_Example)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include(FetchContent)
FetchContent_Declare(
    screen_capture_lite
    GIT_REPOSITORY https://github.com/smasherprog/screen_capture_lite
    GIT_TAG        17.1.2745
)
set(BUILD_EXAMPLE OFF CACHE INTERNAL "Disable screen_capture_lite examples" FORCE)
set(BUILD_CSHARP OFF CACHE INTERNAL "Disable screen_capture_lite C# examples" FORCE)
set(BUILD_SHARED_LIBS ON CACHE INTERNAL "Enable screen_capture_lite shared libraries" FORCE)
FetchContent_MakeAvailable(screen_capture_lite)

file(GLOB SOURCE_FILES "src/examples/*.cpp")

foreach(SOURCE_FILE ${SOURCE_FILES})
    get_filename_component(FILE_NAME ${SOURCE_FILE} NAME_WE)

    set(TARGET_NAME "opencv_${FILE_NAME}")

    add_executable(${TARGET_NAME} ${SOURCE_FILE})

    target_include_directories(${TARGET_NAME} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${screen_capture_lite_SOURCE_DIR}/include
    )

    target_link_libraries(${TARGET_NAME} PRIVATE
        ${OpenCV_LIBS}
        screen_capture_lite_shared
    )
endforeach()

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data"
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")