src_dir = "src"
obj_dir = "obj"
dep_dir = "dep"
bin_dir = "bin"

function read_dep(filename)
    local file = io.open(filename)
    if not file then
        return
    else
        local includes = {}
        local first_line_skipped = false

        for line in file:lines() do
            if first_line_skipped then
                line = line:gsub("[ ]", "")
                line = line:gsub("\\", "")
                local is_include = "src" == string.sub(line, 1, string.len(src_dir))
                if is_include then
                    table.insert(includes, line)
                end
            end

            first_line_skipped = true
        end

        file:close()

        return includes
    end
end

AddTool(function(s)
    s.cc.flags:Add("-Wall")
    s.cc.flags:Add("-Wextra")
    s.cc.flags_cxx:Add("--std=c++20")
    s.cc.includes:Add(src_dir)
    s.cu = {}
    s.cu.flags_compile = {}
    table.insert(s.cu.flags_compile, "-diag-suppress 20012") -- glm
    table.insert(s.cu.flags_compile, "--std=c++20")
    table.insert(s.cu.flags_compile, "--expt-relaxed-constexpr")
    table.insert(s.cu.flags_compile, "--extended-lambda")
    -- table.insert(s.cu.flags_compile, "--ptxas-options=-v")
    s.cu.flags_link = {}
    table.insert(s.cu.flags_link, "GL")
    table.insert(s.cu.flags_link, "GLEW")
    table.insert(s.cu.flags_link, "glfw")
    table.insert(s.cu.flags_link, "jack")
    table.insert(s.cu.flags_link, "cufftw")
    s.cu.flags = {}
    s.name = "default"

    s.cc.Output = function(s, input)
        input = input:gsub("^"..src_dir.."/", "")
        return PathJoin(obj_dir, PathBase(input))
    end

    s.compile.mappings.cu = function(s, input)
        local output = input:gsub("^"..src_dir.."/", "")
        local dep = PathJoin(dep_dir, PathBase(output))..".dep"
        output = PathJoin(obj_dir, PathBase(output))..".o"

        local flags = table.concat(TableFlatten({s.cu.flags_compile, s.cu.flags}), " ")
        AddJob(
            output,
            "nvcc "..input,
            "nvcc "..flags.." -c -o "..output.." "..input
        )
        AddDependency(output, input)

        AddJob(
            dep,
            "nvcc -MM "..input,
            "nvcc "..flags.." -MM "..input.." > "..dep
        )
        AddDependency(dep, input)
        AddDependency(output, dep)
        local includes = read_dep(dep)
        if includes then
            AddDependency(output, includes)
            AddDependency(dep, includes)
        end

        return output
    end

    s.link.exe = "/opt/cuda/bin/nvccxxxx"
    s.link.Driver = function(s, output, inputs)
        print(s)
        print(output)
        print(inputs)
    end
end)

function link(s, output, inputs)
    local flags = table.concat(TableFlatten({s.cu.flags, s.cu.flags_compile, s.cu.flags, s.cu.flags_cxx}), " ")
    local link_flags = ""
    for _, v in ipairs(s.cu.flags_link) do
        link_flags = link_flags.." -l"..v
    end
    local input = table.concat(inputs, " ")
    AddJob(
        output,
        "Linking "..output,
        "nvcc "..flags.." "..link_flags.." -o "..output.." "..input
    )
    AddDependency(output, inputs)
    return output
end

local s = NewSettings()
s.cc.flags:Add("-O3")
table.insert(s.cu.flags, "--optimize 3")

src_cpp = CollectRecursive(PathJoin(src_dir, "*.cpp"))
src_cu = CollectRecursive(PathJoin(src_dir, "*.cu"))

obj_cpp = Compile(s, src_cpp)
obj_cu = Compile(s, src_cu)

obj = TableFlatten({obj_cpp, obj_cu})
bin = PathJoin(bin_dir, "pathtracer")
bin = link(s, bin, obj)

-- target run
AddJob(
    "r",
    "Running '"..bin.."'",
    "./"..bin
)
AddDependency("r", bin)

-- target prof
AddJob(
    "prof",
    "Profiling '"..bin.."'",
    "nvprof "..bin
)
AddDependency("prof", bin)

-- by default only compile
DefaultTarget(bin)
