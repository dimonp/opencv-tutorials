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

set(EXECUTABLES
    "opencv_example" "src/examples/example.cpp"
    "opencv_utility_classes" "src/examples/utility_classes.cpp"
    "opencv_draw_primitives" "src/examples/draw_primitives.cpp"
    "opencv_video_capture" "src/examples/video_capture.cpp"
    "opencv_video_modify" "src/examples/video_modify.cpp"
    "opencv_image_area" "src/examples/image_area.cpp"
    "opencv_image_contour" "src/examples/image_contour.cpp"
    "opencv_image_transform" "src/examples/image_transform.cpp"
    "opencv_screen_capture" "src/examples/screen_capture.cpp"
    "opencv_screen_capture_stream" "src/examples/screen_capture_stream.cpp"
    "opencv_window_capture_stream" "src/examples/window_capture_stream.cpp"
    "opencv_accelartion_opencl" "src/examples/accelartion_opencl.cpp"
    "opencv_dnn_yolo26_onnx_print" "src/examples/dnn_yolo8_onnx_print.cpp"
    "opencv_dnn_yolo8_onnx" "src/examples/dnn_yolo8_onnx.cpp"
    "opencv_dnn_yolo26_onnx" "src/examples/dnn_yolo26_onnx.cpp"
    "opencv_dnn_yolo26_onnx_seg" "src/examples/dnn_yolo26_onnx_seg.cpp"
)

while(EXECUTABLES)
    list(GET EXECUTABLES 0 TARGET_NAME)
    list(GET EXECUTABLES 1 SOURCE_FILE)
    list(REMOVE_AT EXECUTABLES 0 1)

    add_executable(${TARGET_NAME} ${SOURCE_FILE})

    target_include_directories(${TARGET_NAME} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${screen_capture_lite_SOURCE_DIR}/include
    )

    target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBS})
    target_link_libraries(${TARGET_NAME} PRIVATE
        screen_capture_lite_shared
    )
endwhile()

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data"
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")