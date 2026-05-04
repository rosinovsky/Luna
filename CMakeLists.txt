cmake_minimum_required(VERSION 3.16)
project(clinker_forecast VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Опции сборки
option(BUILD_SHARED "Build shared library" ON)
option(BUILD_STATIC "Build static library" ON)
option(BUILD_CLI "Build CLI tool" ON)
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_OPTIMIZATIONS "Enable aggressive optimizations" ON)
option(ENABLE_OPENMP "Enable OpenMP parallelization" OFF)
option(ENABLE_NEON "Enable ARM NEON optimizations" OFF)
option(ENABLE_AVX "Enable AVX/AVX2 optimizations" OFF)

# Компиляторные флаги
if(ENABLE_OPTIMIZATIONS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-O3 -ffast-math -funroll-loops -fomit-frame-pointer)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            add_compile_options(-march=native)
        endif()
    elseif(MSVC)
        add_compile_options(/O2 /fp:fast)
    endif()
endif()

if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_compile_options(${OpenMP_CXX_FLAGS})
        link_libraries(${OpenMP_CXX_LIBRARIES})
    endif()
endif()

# Источники ядра
set(CORE_SOURCES
    core/src/model.cpp
    core/src/inference.cpp
    core/src/export.cpp
    core/src/physics.cpp
    core/src/ops/ops.cpp
)

set(CORE_HEADERS
    core/include/clinker_forecast.h
)

# Объектная библиотека (для переиспользования)
add_library(clinker_core OBJECT ${CORE_SOURCES})
target_include_directories(clinker_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
)

# Shared library
if(BUILD_SHARED)
    add_library(clinker_forecast SHARED $<TARGET_OBJECTS:clinker_core>)
    set_target_properties(clinker_forecast PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        PUBLIC_HEADER "${CORE_HEADERS}"
    )
    target_include_directories(clinker_forecast PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core/include>
        $<INSTALL_INTERFACE:include>
    )
endif()

# Static library
if(BUILD_STATIC)
    add_library(clinker_forecast_static STATIC $<TARGET_OBJECTS:clinker_core>)
    set_target_properties(clinker_forecast_static PROPERTIES
        OUTPUT_NAME clinker_forecast
    )
    target_include_directories(clinker_forecast_static PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core/include>
        $<INSTALL_INTERFACE:include>
    )
endif()

# CLI утилита
if(BUILD_CLI)
    add_executable(clinker_cli tools/cli/main.cpp)
    target_link_libraries(clinker_cli PRIVATE clinker_forecast_static)
    set_target_properties(clinker_cli PROPERTIES
        OUTPUT_NAME clinker_forecast
    )
endif()

# Тесты
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_load tests/test_load.cpp)
    target_link_libraries(test_load PRIVATE clinker_forecast_static)
    add_test(NAME load_model COMMAND test_load)
endif()

# Установка
include(GNUInstallDirs)

if(BUILD_SHARED)
    install(TARGETS clinker_forecast
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

if(BUILD_STATIC)
    install(TARGETS clinker_forecast_static
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

if(BUILD_CLI)
    install(TARGETS clinker_cli
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# CPack для пакетов
set(CPACK_PACKAGE_NAME "clinker-forecast")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Neural forecasting engine for cement clinker quality")
set(CPACK_PACKAGE_VENDOR "ClinkerForecast Team")
include(CPack)

# Сообщение
message(STATUS "")
message(STATUS "ClinkerForecast Core Configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Shared lib:     ${BUILD_SHARED}")
message(STATUS "  Static lib:     ${BUILD_STATIC}")
message(STATUS "  CLI tool:       ${BUILD_CLI}")
message(STATUS "  Tests:          ${BUILD_TESTS}")
message(STATUS "  Optimizations:  ${ENABLE_OPTIMIZATIONS}")
message(STATUS "  OpenMP:         ${ENABLE_OPENMP}")
message(STATUS "")