package com.example.foodorder.fooddroid.Common;

import android.content.Context;
import android.graphics.Typeface;

import com.example.foodorder.fooddroid.Model.User;

public class Common {

    public static User CurrentUser;

    public static Typeface getTypeFace(Context parentContext){

        return Typeface.createFromAsset(parentContext.getAssets(),"fonts/DroidSans.ttf");

    }

}
