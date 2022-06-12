#!/bin/bash
cd ./wheelhouse
for whl in *win*.whl; do
  wheel unpack $whl --dest unpacked
  mv unpacked/*/correlation_integral/.libs/* unpacked/*/
  wheel pack unpacked/*
  rm -rf unpacked
done
cd ../

