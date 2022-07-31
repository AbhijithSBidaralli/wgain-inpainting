FOLDERNAME="inpainting"

cd $FOLDERNAME
zip -r ../$FOLDERNAME.zip * -x "*datasets*" "*saved_models_celeba*" "*.log"
cd ..
echo '#!/usr/bin/env python3' | cat - $FOLDERNAME.zip > $FOLDERNAME.exe
chmod a+x $FOLDERNAME.exe
rm $FOLDERNAME.zip

echo "Done"