# conda remove -n phmr --all -y
# conda create -n phmr python=3.11.9 -y
# conda activate phmr

# conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y

# Parse command line arguments
PT_VERSION=""
WORLD_VIDEO="false"  # Default to false

show_usage() {
    echo "Usage: $0 --pt_version <version> [--world-video=<true|false>]"
    echo ""
    echo "Options:"
    echo "  --pt_version <version>       PyTorch version to install (2.4 or 2.6)"
    echo "  --world-video <true|false>   Download data and install wheel packages (default: false)"
    echo "  --help                       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --pt_version=2.4"
    echo "  $0 --pt_version=2.6"
    echo "  $0 --pt_version=2.4 --world-video=true"
    echo "  $0 --pt_version=2.6 --world-video=false"
}

# Function to parse boolean values
parse_boolean() {
    local value="$1"
    case "${value,,}" in  # Convert to lowercase
        true|yes|1|on)
            echo "true"
            ;;
        false|no|0|off)
            echo "false"
            ;;
        *)
            echo "Error: Invalid boolean value '$value'. Use true/false, yes/no, 1/0, or on/off"
            show_usage
            exit 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --pt_version)
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                PT_VERSION="$2"
                shift 2
            else
                echo "Error: --pt_version requires a value"
                show_usage
                exit 1
            fi
            ;;
        --pt_version=*)
            PT_VERSION="${1#*=}"
            shift
            ;;
        --world-video)
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                WORLD_VIDEO=$(parse_boolean "$2")
                shift 2
            else
                echo "Error: --world-video requires a value"
                show_usage
                exit 1
            fi
            ;;
        --world-video=*)
            WORLD_VIDEO=$(parse_boolean "${1#*=}")
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            show_usage
            exit 1
            ;;
    esac
done

# Validation check for PyTorch version argument
if [ -z "$PT_VERSION" ]; then
    echo "Error: PyTorch version argument is required."
    show_usage
    exit 1
fi

if [ "$PT_VERSION" != "2.4" ] && [ "$PT_VERSION" != "2.6" ]; then
    echo "Error: Invalid PyTorch version '$PT_VERSION'."
    echo "Supported versions: 2.4, 2.6"
    show_usage
    exit 1
fi

echo "Installing PyTorch version $PT_VERSION..."

eval "$(conda shell.bash hook)"

if [ "$WORLD_VIDEO" == "true" ]; then
    echo "World-video option enabled: Will download data and install wheel packages"
else
    echo "World-video option disabled: Skipping data download and wheel installation"
fi

if [ "$PT_VERSION" == "2.4" ]; then
    conda create -n phmr_pt$PT_VERSION python=3.11.9 -y
    conda activate phmr_pt$PT_VERSION

    echo "Installing PyTorch 2.4 and compatible packages..."
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    pip install --upgrade setuptools pip
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
    conda install -c conda-forge suitesparse -y

    pip install -r requirements.txt 

    pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 --no-deps

    if [ "$WORLD_VIDEO" == "true" ]; then
        gdown --folder -O ./data/ https://drive.google.com/drive/folders/1IXyhVqL25ofI-tYqyUZCqF-h4V20795H?usp=sharing

        pip install data/wheels/detectron2-0.8-cp311-cp311-linux_x86_64.whl
        pip install data/wheels/droid_backends_intr-0.3-cp311-cp311-linux_x86_64.whl
        pip install data/wheels/lietorch-0.3-cp311-cp311-linux_x86_64.whl
        pip install data/wheels/sam2-1.5-cp311-cp311-linux_x86_64.whl
        pip install data/wheels/gloss-0.5.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    fi

elif [ "$PT_VERSION" == "2.6" ]; then
    echo "Installing PyTorch 2.6 and compatible packages..."
    conda create -n phmr_pt$PT_VERSION python=3.12.9 -y
    conda activate phmr_pt$PT_VERSION

    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

    pip install --upgrade setuptools pip
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

    pip install -r requirements.txt

    pip install -U xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu126 --no-deps

    if [ "$WORLD_VIDEO" == "true" ]; then
        gdown --folder -O ./data/ https://drive.google.com/drive/folders/151gPvMaUWok_pDQT6h8Rpvk_rCcKvcWZ?usp=sharing

        pip install data/wheels/sam2-1.6-cp312-cp312-linux_x86_64.whl
        pip install data/wheels/detectron2-0.9-cp312-cp312-linux_x86_64.whl
        pip install data/wheels/droid_backends_intr-0.4-cp312-cp312-linux_x86_64.whl
        # pip install data/wheels/gloss_rs-0.6.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
        pip install data/wheels/gloss-0.5.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
        pip install data/wheels/lietorch-0.4-cp312-cp312-linux_x86_64.whl
    fi
fi

echo "Installation completed for PyTorch $PT_VERSION!"



