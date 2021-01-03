var cubeStrip = [
    1, 1, 0,
    0, 1, 0,
    1, 1, 1,
    0, 1, 1,
    0, 0, 1,
    0, 1, 0,
    0, 0, 0,
    1, 1, 0,
    1, 0, 0,
    1, 1, 1,
    1, 0, 1,
    0, 0, 1,
    1, 0, 0,
    0, 0, 0
];

var colorMap = [[13, 8, 135, 255], [16, 7, 136, 255], [19, 7, 137, 255], [25, 6, 140, 255], [27, 6, 141, 255], [32, 6, 143, 255], [34, 6, 144, 255], [38, 5, 145, 255], [40, 5, 146, 255], [42, 5, 147, 255], [46, 5, 149, 255], [47, 5, 150, 255], [51, 5, 151, 255], [53, 4, 152, 255], [56, 4, 154, 255], [58, 4, 154, 255], [60, 4, 155, 255], [63, 4, 156, 255], [65, 4, 157, 255], [68, 3, 158, 255], [70, 3, 159, 255], [73, 3, 160, 255], [75, 3, 161, 255], [76, 2, 161, 255], [80, 2, 162, 255], [81, 2, 163, 255], [85, 2, 164, 255], [86, 1, 164, 255], [89, 1, 165, 255], [91, 1, 165, 255], [92, 1, 166, 255], [96, 1, 166, 255], [97, 0, 167, 255], [100, 0, 167, 255], [102, 0, 167, 255], [105, 0, 168, 255], [107, 0, 168, 255], [108, 0, 168, 255], [111, 0, 168, 255], [113, 0, 168, 255], [116, 1, 168, 255], [117, 1, 168, 255], [120, 1, 168, 255], [122, 2, 168, 255], [123, 2, 168, 255], [126, 3, 168, 255], [128, 4, 168, 255], [131, 5, 167, 255], [132, 6, 167, 255], [135, 7, 166, 255], [136, 8, 166, 255], [138, 9, 165, 255], [141, 11, 165, 255], [142, 12, 164, 255], [145, 14, 163, 255], [146, 15, 163, 255], [149, 17, 161, 255], [150, 19, 161, 255], [152, 20, 160, 255], [154, 22, 159, 255], [156, 23, 158, 255], [158, 25, 157, 255], [160, 26, 156, 255], [162, 29, 154, 255], [163, 30, 154, 255], [165, 31, 153, 255], [167, 33, 151, 255], [168, 34, 150, 255], [171, 36, 148, 255], [172, 38, 148, 255], [174, 40, 146, 255], [176, 41, 145, 255], [177, 42, 144, 255], [179, 44, 142, 255], [180, 46, 141, 255], [182, 48, 139, 255], [183, 49, 138, 255], [186, 51, 137, 255], [187, 52, 136, 255], [188, 53, 135, 255], [190, 56, 133, 255], [191, 57, 132, 255], [193, 59, 130, 255], [194, 60, 129, 255], [196, 62, 127, 255], [197, 64, 126, 255], [198, 65, 125, 255], [200, 67, 123, 255], [201, 68, 122, 255], [203, 70, 121, 255], [204, 71, 120, 255], [205, 74, 118, 255], [206, 75, 117, 255], [208, 77, 115, 255], [209, 78, 114, 255], [210, 79, 113, 255], [212, 82, 112, 255], [213, 83, 111, 255], [214, 85, 109, 255], [215, 86, 108, 255], [217, 89, 106, 255], [218, 90, 106, 255], [218, 91, 105, 255], [220, 93, 103, 255], [221, 94, 102, 255], [222, 97, 100, 255], [223, 98, 99, 255], [225, 100, 98, 255], [226, 101, 97, 255], [226, 102, 96, 255], [228, 105, 94, 255], [229, 106, 93, 255], [230, 108, 92, 255], [231, 110, 91, 255], [232, 112, 89, 255], [233, 113, 88, 255], [233, 114, 87, 255], [235, 117, 86, 255], [235, 118, 85, 255], [237, 121, 83, 255], [237, 122, 82, 255], [239, 124, 81, 255], [239, 126, 80, 255], [240, 127, 79, 255], [241, 129, 77, 255], [241, 131, 76, 255], [243, 133, 75, 255], [243, 135, 74, 255], [244, 137, 72, 255], [245, 139, 71, 255], [245, 140, 70, 255], [246, 143, 68, 255], [247, 144, 68, 255], [247, 147, 66, 255], [248, 148, 65, 255], [249, 151, 63, 255], [249, 152, 62, 255], [249, 154, 62, 255], [250, 156, 60, 255], [250, 158, 59, 255], [251, 161, 57, 255], [251, 162, 56, 255], [252, 165, 55, 255], [252, 166, 54, 255], [252, 168, 53, 255], [253, 171, 51, 255], [253, 172, 51, 255], [253, 175, 49, 255], [253, 177, 48, 255], [253, 180, 47, 255], [253, 181, 46, 255], [254, 183, 45, 255], [254, 186, 44, 255], [254, 187, 43, 255], [254, 190, 42, 255], [254, 192, 41, 255], [253, 195, 40, 255], [253, 197, 39, 255], [253, 198, 39, 255], [253, 202, 38, 255], [253, 203, 38, 255], [252, 206, 37, 255], [252, 208, 37, 255], [251, 211, 36, 255], [251, 213, 36, 255], [251, 215, 36, 255], [250, 218, 36, 255], [249, 220, 36, 255], [248, 223, 37, 255], [248, 225, 37, 255], [247, 228, 37, 255], [246, 230, 38, 255], [246, 232, 38, 255], [245, 235, 39, 255], [244, 237, 39, 255], [243, 240, 39, 255], [242, 242, 39, 255], [241, 245, 37, 255], [240, 247, 36, 255], [240, 249, 34, 255]];

var takeScreenShot = false;
var canvas = null;

var device = null;
var adapter = null;
var dataBuf = null;
var volumeDataBuffer = null;
var volumeParamsBuffer = null;
var viewParamsBuffer = null;
var colorMapBuffer = null;
var renderPipeline = null;
var renderPassDesc = null;
var swapChain = null;
var fileRegex = /.*\/(\w+)_(\d+)x(\d+)x(\d+)_(\w+)\.*/;
var proj = null;
var camera = null;
var projView = null;
var tabFocused = true;
var newVolumeUpload = true;
var targetFrameTime = 32;
var samplingRate = 1.0;
var WIDTH = 640;
var HEIGHT = 480;

const defaultEye = vec3.set(vec3.create(), 0.5, 0.5, 1.5);
const center = vec3.set(vec3.create(), 0.5, 0.5, 0.5);
const up = vec3.set(vec3.create(), 0.0, 1.0, 0.0);

var volumes = {
    "Fuel": "7d87jcsh0qodk78/fuel_64x64x64_uint8.raw",
    "Neghip": "zgocya7h33nltu9/neghip_64x64x64_uint8.raw",
    "Hydrogen Atom": "jwbav8s3wmmxd5x/hydrogen_atom_128x128x128_uint8.raw",
    "Boston Teapot": "w4y88hlf2nbduiv/boston_teapot_256x256x178_uint8.raw",
    "Engine": "ld2sqwwd3vaq4zf/engine_256x256x128_uint8.raw",
    "Bonsai": "rdnhdxmxtfxe0sa/bonsai_256x256x256_uint8.raw",
    "Foot": "ic0mik3qv4vqacm/foot_256x256x256_uint8.raw",
    "Skull": "5rfjobn0lvb7tmo/skull_256x256x256_uint8.raw",
    "Aneurysm": "3ykigaiym8uiwbp/aneurism_256x256x256_uint8.raw",
};

var colormaps = {
    "Cool Warm": "colormaps/cool-warm-paraview.png",
    "Matplotlib Plasma": "colormaps/matplotlib-plasma.png",
    "Matplotlib Virdis": "colormaps/matplotlib-virdis.png",
    "Rainbow": "colormaps/rainbow.png",
    "Samsel Linear Green": "colormaps/samsel-linear-green.png",
    "Samsel Linear YGB 1211G": "colormaps/samsel-linear-ygb-1211g.png",
};

var loadVolume = function (file, onload) {
    var m = file.match(fileRegex);
    var volDims = [parseInt(m[2]), parseInt(m[3]), parseInt(m[4])];

    var url = "https://www.dl.dropboxusercontent.com/s/" + file + "?dl=1";
    var req = new XMLHttpRequest();
    var loadingProgressText = document.getElementById("loadingText");
    var loadingProgressBar = document.getElementById("loadingProgressBar");

    loadingProgressText.innerHTML = "Loading Volume";
    loadingProgressBar.setAttribute("style", "width: 0%");

    req.open("GET", url, true);
    req.responseType = "arraybuffer";
    req.onprogress = function (evt) {
        var vol_size = volDims[0] * volDims[1] * volDims[2];
        var percent = evt.loaded / vol_size * 100;
        loadingProgressBar.setAttribute("style", "width: " + percent.toFixed(2) + "%");
    };
    req.onerror = function (evt) {
        loadingProgressText.innerHTML = "Error Loading Volume";
        loadingProgressBar.setAttribute("style", "width: 0%");
    };
    req.onload = function (evt) {
        loadingProgressText.innerHTML = "Loaded Volume";
        loadingProgressBar.setAttribute("style", "width: 100%");
        var dataBuffer = req.response;
        if (dataBuffer) {
            dataBuffer = new Uint8Array(dataBuffer);
            onload(file, dataBuffer);
        } else {
            alert("Unable to load buffer properly from volume?");
            console.log("no buffer?");
        }
    };
    req.send();
}

var selectVolume = function () {
    var selection = document.getElementById("volumeList").value;
    history.replaceState(history.state, "#" + selection, "#" + selection);

    loadVolume(volumes[selection], function (file, dataBuffer) {
        var m = file.match(fileRegex);
        var volDims = [parseFloat(m[2]), parseFloat(m[3]), parseFloat(m[4])];

        var longestAxis = Math.max(volDims[0], Math.max(volDims[1], volDims[2]));
        var volScale = [volDims[0] / longestAxis, volDims[1] / longestAxis,
        volDims[2] / longestAxis];
        // Upload the volume data
        upload = device.createBuffer({
            size: 64 * 64 * 64 * 1,
            usage: GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true
        })
        new Uint8Array(upload.getMappedRange()).set(dataBuffer);
        upload.unmap();

        var commandEncoder = device.createCommandEncoder();

        commandEncoder.copyBufferToBuffer(upload, 0, volumeDataBuffer, 0, 64 * 64 * 64 * 1);

        newVolumeUpload = true;
        setInterval(function () {
            // Save them some battery if they're not viewing the tab
            if (document.hidden) {
                return;
            }

            // Reset the sampling rate and camera for new volumes
            if (newVolumeUpload) {
                camera = new ArcballCamera(defaultEye, center, up, 2, [WIDTH, HEIGHT]);
            }
            renderPassDesc.colorAttachments[0].attachment = swapChain.getCurrentTexture().createView();

            // Compute and upload the combined projection and view matrix
            projView = mat4.mul(projView, proj, camera.camera);
            var eye = [camera.invCamera[12], camera.invCamera[13], camera.invCamera[14]];
            var upload = device.createBuffer({
                size: 16 * 4 + 2 * 3 * 4,
                usage: GPUBufferUsage.COPY_SRC,
                mappedAtCreation: true
            });
            var map = new Float32Array(upload.getMappedRange());
            map.set(projView);
            map.set(eye, 16);
            map.set(volScale, 19);
            upload.unmap();

            // Compute and upload the volume params
            var test = device.createBuffer({
                size: 3 * 4 + 4,
                usage: GPUBufferUsage.COPY_SRC,
                mappedAtCreation: true
            })
            // TODO have to separate volume dims because float vs int
            var map = new Float32Array(test.getMappedRange());
            map.set(volDims);
            map.set([samplingRate], 3);
            upload.unmap();

            var commandEncoder = device.createCommandEncoder();

            commandEncoder.copyBufferToBuffer(upload, 0, viewParamsBuffer, 0, 16 * 4 + 2 * 3 * 4);
            commandEncoder.copyBufferToBuffer(upload, 0, volumeParamsBuffer, 0, 3 * 4 + 4);

            var renderPass = commandEncoder.beginRenderPass(renderPassDesc);

            renderPass.setPipeline(renderPipeline);
            renderPass.setVertexBuffer(0, dataBuf);
            renderPass.draw(cubeStrip.length / 3, 1, 0, 0);

            renderPass.endPass();
            device.defaultQueue.submit([commandEncoder.finish()]);

            newVolumeUpload = false;
        }, targetFrameTime);
    });
}

var selectColormap = function () {
    var selection = document.getElementById("colormapList").value;
    var colormapImage = new Image();
    colormapImage.onload = function () {
        gl.activeTexture(gl.TEXTURE1);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 180, 1,
            gl.RGBA, gl.UNSIGNED_BYTE, colormapImage);
    };
    colormapImage.src = colormaps[selection];
}

var fillVolumeSelector = function () {
    var selector = document.getElementById("volumeList");
    for (v in volumes) {
        var opt = document.createElement("option");
        opt.value = v;
        opt.innerHTML = v;
        selector.appendChild(opt);
    }
}

var fillcolormapSelector = function () {
    var selector = document.getElementById("colormapList");
    for (p in colormaps) {
        var opt = document.createElement("option");
        opt.value = p;
        opt.innerHTML = p;
        selector.appendChild(opt);
    }
}

window.onload = async function () {
    fillVolumeSelector();
    fillcolormapSelector();

    if (!navigator.gpu) {
        alert("WebGPU is not supported/enabled in your browser");
        return;
    }

    // Get a GPU device to render with
    adapter = await navigator.gpu.requestAdapter();
    device = await adapter.requestDevice();

    // Get a context to display our rendered image on the canvas
    var canvas = document.getElementById("webgpu-canvas");
    var context = canvas.getContext("gpupresent");

    // Setup camera controls
    proj = mat4.perspective(mat4.create(), 60 * Math.PI / 180.0,
        canvas.width / canvas.height, 0.1, 100);

    camera = new ArcballCamera(defaultEye, center, up, 2, [WIDTH, HEIGHT]);
    projView = mat4.create();

    // Register mouse and touch listeners
    var controller = new Controller();
    controller.mousemove = function (prev, cur, evt) {
        if (evt.buttons == 1) {
            camera.rotate(prev, cur);

        } else if (evt.buttons == 2) {
            camera.pan([cur[0] - prev[0], prev[1] - cur[1]]);
        }
    };
    controller.wheel = function (amt) { camera.zoom(amt); };
    controller.pinch = controller.wheel;
    controller.twoFingerDrag = function (drag) { camera.pan(drag); };

    document.addEventListener("keydown", function (evt) {
        if (evt.key == "p") {
            takeScreenShot = true;
        }
    });

    controller.registerForCanvas(canvas);

    var vertModule = device.createShaderModule({ code: simple_vert_spv });
    var vertexStage = {
        module: vertModule,
        entryPoint: "main"
    };

    var fragModule = device.createShaderModule({ code: simple_frag_spv });
    var fragmentStage = {
        module: fragModule,
        entryPoint: "main"
    };

    // Specify vertex data
    dataBuf = device.createBuffer({
        size: 14 * 3 * 4,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    // Vertex positions
    new Float32Array(dataBuf.getMappedRange()).set(cubeStrip);
    dataBuf.unmap();

    var vertexState = {
        vertexBuffers: [
            {
                arrayStride: 3 * 4,
                attributes: [
                    {
                        format: "float3",
                        offset: 0,
                        shaderLocation: 0
                    }
                ]
            }
        ]
    };

    // Setup render outputs
    var swapChainFormat = "bgra8unorm";
    swapChain = context.configureSwapChain({
        device: device,
        format: swapChainFormat,
        usage: GPUTextureUsage.OUTPUT_ATTACHMENT
    });

    var depthFormat = "depth24plus-stencil8";
    var depthTexture = device.createTexture({
        size: {
            width: canvas.width,
            height: canvas.height,
            depth: 1
        },
        format: depthFormat,
        usage: GPUTextureUsage.OUTPUT_ATTACHMENT
    });

    // Create bind group layout
    var bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                // One or more stage flags, or'd together
                visibility: GPUShaderStage.VERTEX,
                type: "uniform-buffer"
            },
            {
                binding: 1,
                // One or more stage flags, or'd together
                visibility: GPUShaderStage.FRAGMENT,
                type: "storage-buffer"
            },
            {
                binding: 2,
                // One or more stage flags, or'd together
                visibility: GPUShaderStage.FRAGMENT,
                type: "uniform-buffer"
            },
            {
                binding: 3,
                // One or more stage flags, or'd together
                visibility: GPUShaderStage.FRAGMENT,
                type: "uniform-buffer"
            },
        ]
    });

    // Create the pipeline layout, specifying which bind group layouts will be used
    var layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    // Create a buffer to store the volume data
    volumeDataBuffer = device.createBuffer({
        size: 64 * 64 * 64 * 1,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    // Create a buffer to store the selected color maps
    colorMapBuffer = device.createBuffer({
        size: 180 * 1 * 4 * 1,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Create a buffer to store the volume parameters
    volumeParamsBuffer = device.createBuffer({
        size: 3 * 4 + 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Create a buffer to store the view parameters
    viewParamsBuffer = device.createBuffer({
        size: 16 * 4 + 2 * 3 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Create a bind group which places our view params buffer at binding 0
    bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: viewParamsBuffer
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: volumeDataBuffer
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: colorMapBuffer
                }
            },
            {
                binding: 3,
                resource: {
                    buffer: volumeParamsBuffer
                }
            },
        ]
    });

    // Create render pipeline
    renderPipeline = device.createRenderPipeline({
        layout: layout,
        vertexStage: vertexStage,
        fragmentStage: fragmentStage,
        primitiveTopology: "triangle-list",
        vertexState: vertexState,
        colorStates: [{
            format: swapChainFormat
        }],
        depthStencilState: {
            format: depthFormat,
            depthWriteEnabled: true,
            depthCompare: "less"
        }
    });

    renderPassDesc = {
        colorAttachments: [{
            attachment: undefined,
            loadValue: [0.3, 0.3, 0.3, 1]
        }],
        depthStencilAttachment: {
            attachment: depthTexture.createView(),
            depthLoadValue: 1.0,
            depthStoreOp: "store",
            stencilLoadValue: 0,
            stencilStoreOp: "store"
        }
    };

    // Load the default colormap and upload it, after which we
    // load the default volume.
    var colormapImage = new Image();
    colormapImage.onload = async function () {
        // Compute and upload the colormap
        upload = device.createBuffer({
            size: 180 * 4,
            usage: GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true
        })
        new Uint8Array(upload.getMappedRange()).set(colorMap);
        upload.unmap();

        var commandEncoder = device.createCommandEncoder();

        commandEncoder.copyBufferToBuffer(upload, 0, colorMapBuffer, 0, 180 * 4);

        selectVolume();
    };
    colormapImage.src = "colormaps/cool-warm-paraview.png";
    //TODO add the functionality to not render when canvas is off screen
}