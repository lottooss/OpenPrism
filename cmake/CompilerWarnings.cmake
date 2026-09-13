# cmake/CompilerWarnings.cmake
# Strict compiler warning policy treating warnings as errors for project code

function(aim_set_compiler_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            /W4          # Baseline warning level 4
            /WX          # Treat warnings as errors
            /permissive- # Strict standards conformance
            /Zc:preprocessor # Standards-conforming preprocessor
            /wd4127      # conditional expression is constant (in test assertion macros comparing constants)
            /wd4324      # structure was padded due to alignment specifier (intentional SIMD alignment)
            /wd5105      # Windows SDK winbase.h macro expansion producing 'defined' warning with conforming preprocessor
            /w14242      # 'identifier': conversion from 'type1' to 'type2', possible loss of data
            /w14254      # 'operator': conversion from 'type1:field_bits' to 'type2:field_bits'
            /w14263      # 'function': member function does not override any base class virtual member function
            /w14265      # 'class': class has virtual functions, but destructor is not virtual
            /w14287      # 'operator': unsigned/negative constant mismatch
            /we4289      # nonstandard extension used: 'variable': loop control variable declared in the for-loop is used outside
            /w14296      # 'operator': expression is always 'boolean_value'
            /w14311      # 'variable': pointer truncation from 'type1' to 'type2'
            /w14545      # expression before comma evaluates to a function which is missing an argument list
            /w14546      # function call before comma missing argument list
            /w14547      # 'operator': operator before comma has no effect; expected operator with side-effect
            /w14549      # 'operator': operator before comma has no effect; did you intend 'operator'?
            /w14555      # expression has no effect; expected expression with side-effect
            /w14619      # pragma warning: there is no warning number 'number'
            /w14640      # Enable warning on thread un-safe static member initialization
            /w14826      # Conversion from 'type1' to 'type_2' is sign-extended
            /w14905      # wide string literal cast to 'LPSTR'
            /w14906      # string literal cast to 'LPWSTR'
            /w14928      # illegal copy-initialization; more than one user-defined conversion has been implicitly applied
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        target_compile_options(${target_name} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
            -Wshadow
            -Wnon-virtual-dtor
            -Wold-style-cast
            -Wcast-align
            -Wunused
            -Woverloaded-virtual
            -Wconversion
            -Wsign-conversion
            -Wnull-dereference
            -Wdouble-promotion
            -Wformat=2
        )
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target_name} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
            -Wshadow
            -Wnon-virtual-dtor
            -Wold-style-cast
            -Wcast-align
            -Wunused
            -Woverloaded-virtual
            -Wconversion
            -Wsign-conversion
            -Wnull-dereference
            -Wdouble-promotion
            -Wformat=2
            -Wduplicated-cond
            -Wduplicated-branches
            -Wlogical-op
        )
    endif()
endfunction()
