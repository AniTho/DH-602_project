mkdir -p datasets
cd datasets
mkdir -p kneeOA    
cd kneeOA

wget https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/289dd733-3e9c-491e-b1e7-6b6cb3a58ba5 -O kneeOA.zip
unzip kneeOA.zip

echo "Processing Knee Ostheoarthritis dataset completed"