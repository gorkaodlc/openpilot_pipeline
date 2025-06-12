cmake_minimum_required(VERSION 2.8)

# Package name (will be called "${PCK}.pck" by default)
set(PCK rtmaps_undistort)
project("${PCK}.u")

# Set the RTMaps SDK directory
if (NOT DEFINED RTMAPS_SDKDIR)
    if (DEFINED ENV{RTMAPS_SDKDIR})
        # Use the default RTMaps installation
        set(RTMAPS_SDKDIR "$ENV{RTMAPS_SDKDIR}" CACHE PATH "")
    else()
        set(RTMAPS_SDKDIR "/opt/rtmaps/" CACHE PATH "")
 	message(WARNING "RTMAPS_SDKDIR not defined. Default value set to: /opt/rtmaps/. Change it appropriately if needed")
    endif()
endif()



##########################3

#---------------------------------------------------#
#              OpenCV Configuration                 #
#---------------------------------------------------#
# Automatically find OpenCV files
FIND_PACKAGE( OpenCV REQUIRED )
message("OpenCV include: " ${OpenCV_INCLUDE_DIRS})
message("OpenCV libs: " ${OpenCV_LIBS})
##############################################


#---------------------------------------------------#
#              viulib			                    #
#---------------------------------------------------#
# Automatically find and addition of  viulib files
# Automatically added include paths and lib paths

########################################333

# Include the RTMaps CMake configuration file
include("${RTMAPS_SDKDIR}/templates.u/rtmaps.cmake")


INCLUDE_DIRECTORIES( ./local_interfaces ${OpenCV_INCLUDE_DIRS} )

LINK_DIRECTORIES( ${MAIN_PROJECT_BUILD_PATH} )

## use, i.e. don't skip the full RPATH for the build tree
# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
#SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:$ORIGIN/lib")
SET(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

# Create the pck target
# NB: dir(pckInfoFile)/local_interfaces, dir(pckInfoFile)/protected_interfaces and dir(pckInfoFile)/public_interfaces
#     will be automatically added to the list of include directories
add_rtmaps_package("${PCK}" PCKINFO "./${PCK}.pckinfo"
    # add sources, headers and/or directories
    "./local_interfaces"
    "./src"
)

#############################
# Include library dependencies here
TARGET_LINK_LIBRARIES( ${PCK} ${OpenCV_LIBS})

# set c++14 as the used c++ standard for this library
#DEFINE_CPP_STANDARD_AS_14(${PCK})

