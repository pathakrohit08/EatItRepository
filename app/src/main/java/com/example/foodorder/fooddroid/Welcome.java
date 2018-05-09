package com.example.foodorder.fooddroid;

import android.graphics.Typeface;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.design.widget.BottomNavigationView;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.view.MenuItem;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import com.example.foodorder.fooddroid.Model.Category;
import com.example.foodorder.fooddroid.ViewHolder.MenuAdapter;
import com.mancj.materialsearchbar.MaterialSearchBar;

import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class Welcome extends AppCompatActivity {


    MaterialSearchBar searchBar;
    RecyclerView recycler_menu;
    RecyclerView.LayoutManager layoutManager;
    private List<Category> menuList=new ArrayList<>();
    private MenuAdapter mn;
    private BottomNavigationView.OnNavigationItemSelectedListener mOnNavigationItemSelectedListener
            = new BottomNavigationView.OnNavigationItemSelectedListener() {

        @Override
        public boolean onNavigationItemSelected(@NonNull MenuItem item) {
            switch (item.getItemId()) {
                case R.id.navigation_home:
                    Toast.makeText(Welcome.this, "You clicked Home", Toast.LENGTH_SHORT).show();
                    return true;
                case R.id.navigation_notifications:
                    Toast.makeText(Welcome.this, "You clicked Notifications", Toast.LENGTH_SHORT).show();
                    return true;
                case R.id.navigation_profile:
                    Toast.makeText(Welcome.this, "You clicked Profile", Toast.LENGTH_SHORT).show();
                    return true;
                case R.id.navigation_messages:
                    Toast.makeText(Welcome.this, "You clicked Message", Toast.LENGTH_SHORT).show();
                    return true;
                case R.id.navigation_order:
                    Toast.makeText(Welcome.this, "You clicked Order", Toast.LENGTH_SHORT).show();
                    return true;
            }
            return false;
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_welcome);



        BottomNavigationView navigation = findViewById(R.id.navigation);
        navigation.setOnNavigationItemSelectedListener(mOnNavigationItemSelectedListener);




        recycler_menu=findViewById(R.id.recycler_MENU);
        layoutManager=new LinearLayoutManager(this);
        recycler_menu.setLayoutManager(layoutManager);
        recycler_menu.setHasFixedSize(true);


        mn=new MenuAdapter(menuList,this);
        recycler_menu.setAdapter(mn);
        loadMenu();

        searchBar=findViewById(R.id.searchBox);


        searchBar.setHint("Implementation pending");


    }


    private void loadMenu() {


        Category c=new Category("Cat1","https://wi-images.condecdn.net/image/RmkaWqWXq8G/crop/1620/f/shutterstock_65735200.jpg");
        menuList.add(c);

        c=new Category("Cat2","https://wi-images.condecdn.net/image/RmkaWqWXq8G/crop/1620/f/shutterstock_65735200.jpg");
        menuList.add(c);

        c=new Category("Cat3","https://wi-images.condecdn.net/image/RmkaWqWXq8G/crop/1620/f/shutterstock_65735200.jpg");
        menuList.add(c);

        c=new Category("Cat4","https://wi-images.condecdn.net/image/RmkaWqWXq8G/crop/1620/f/shutterstock_65735200.jpg");
        menuList.add(c);


        mn.notifyDataSetChanged();



    }

}
